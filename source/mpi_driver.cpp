// MPI driver for XnYZip distributed compression.
//
// Following the design discussed for the PVLDB revision: each rank reads
// disjoint fixed-size byte blocks from a shared input via MPI_File_read_at,
// compresses each block independently with the existing XnYZip pipeline,
// writes a rank-local part file + per-block JSONL manifest, and reports
// max-over-rank phase timings via MPI_Reduce.
//
// Block assignment is round-robin: rank r owns blocks {r, r+nranks, r+2*nranks, ...}.
// Compressed output of rank r is concatenated in block-id order into
// part_rank<R>.bin; the manifest records (block_id, input_offset, input_size,
// rank, part_offset, compressed_size) so a downstream tool can reconstruct
// global input order during decompression.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "lib.hpp"

namespace {

constexpr std::uint64_t POINT_BYTES = 12; // float32 * 3

struct Args {
    std::string input;
    std::string output_dir;
    std::uint64_t block_size = 64ull * 1024ull * 1024ull;
    std::string quantizer = "to";   // "cube" | "to"
    std::string curve = "-z";       // "-z" | "-h"
    std::string mode = "-normal";   // "-normal" | "-rle" | "mapreduce"
                                    // "mapreduce" runs Map+Shuffle+Reduce+Encode
                                    // for global RLE dedup across ranks. Each
                                    // rank owns a Morton-prefix range of cells
                                    // and emits ONE part file per rank (not
                                    // per chunk). Output is point-multiset; the
                                    // original input order is NOT preserved.
    float bound = 1e-3f;
    int direct_threshold = 1024;
    bool shared_codebook = false;   // --shared-codebook: warmup pass on rank 0,
                                    // broadcast Huffman codebooks, every chunk
                                    // emits identical codebook bytes
    bool verify_roundtrip = false;  // --verify-roundtrip: decompress each chunk
                                    // immediately after compressing and verify
                                    // every point is within bound. Aborts if not.
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --input PATH --output-dir DIR --block-size BYTES --bound F "
        "[--quantizer cube|to] [--curve -z|-h] [--mode -normal|-rle] "
        "[--direct-threshold N] [--shared-codebook]\n", prog);
}

Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--input") a.input = need("--input");
        else if (k == "--output-dir") a.output_dir = need("--output-dir");
        else if (k == "--block-size") a.block_size = std::stoull(need("--block-size"));
        else if (k == "--bound") a.bound = std::stof(need("--bound"));
        else if (k == "--quantizer") a.quantizer = need("--quantizer");
        else if (k == "--curve") a.curve = need("--curve");
        else if (k == "--mode") a.mode = need("--mode");
        else if (k == "--direct-threshold") a.direct_threshold = std::stoi(need("--direct-threshold"));
        else if (k == "--shared-codebook") a.shared_codebook = true;
        else if (k == "--verify-roundtrip") a.verify_roundtrip = true;
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.input.empty() || a.output_dir.empty()) {
        usage(argv[0]);
        throw std::runtime_error("--input and --output-dir are required");
    }
    if (a.block_size % POINT_BYTES != 0) {
        throw std::runtime_error("block size must be a multiple of 12 bytes (3*float32)");
    }
    if (a.shared_codebook && a.mode == "-rle") {
        throw std::runtime_error("--shared-codebook is not implemented for -rle mode");
    }
    if (a.shared_codebook && a.mode == "mapreduce") {
        throw std::runtime_error("--shared-codebook is N/A for mapreduce mode (global dedup makes per-rank codebook irrelevant)");
    }
    if (a.mode != "-normal" && a.mode != "-rle" && a.mode != "mapreduce") {
        throw std::runtime_error("--mode must be one of: -normal, -rle, mapreduce");
    }
    if (a.mode == "mapreduce" && a.quantizer == "adaptive") {
        throw std::runtime_error("mapreduce mode does not support adaptive quantizer (per-chunk params can't be globalized)");
    }
    return a;
}

std::vector<Eigen::RowVector3f> bytes_to_points(const std::vector<char>& buf) {
    std::vector<Eigen::RowVector3f> out;
    std::uint64_t n = buf.size() / POINT_BYTES;
    out.reserve(n);
    const float* f = reinterpret_cast<const float*>(buf.data());
    for (std::uint64_t i = 0; i < n; ++i) {
        out.emplace_back(f[3*i], f[3*i + 1], f[3*i + 2]);
    }
    return out;
}

std::vector<std::uint8_t> compress_block(std::vector<Eigen::RowVector3f>& pts,
                                         const Args& a,
                                         const XnYZip::BlockEncodingProfile* profile) {
    using namespace XnYZip;
    bool is_hilbert = (a.curve == "-h");
    bool use_rle = (a.mode == "-rle");
    QUANTIZER_TYPE qt = (a.quantizer == "cube") ? QUANTIZER_TYPE::CUBE
                                                : QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
    if (use_rle) {
        // shared-codebook is unsupported for RLE; the parser already rejects
        // this combination, so profile must be null here.
        BlockCompressorRLE<float> c(qt, a.bound, false, a.direct_threshold);
        return c.compress(pts, is_hilbert);
    }
    BlockCompressor<float> c(qt, a.bound, false, a.direct_threshold);
    if (profile != nullptr) {
        return c.compress_with_profile(pts, is_hilbert, *profile);
    }
    return c.compress(pts, is_hilbert);
}

// =================================================================
// mapreduce mode: Map + Shuffle + Reduce + Encode
// =================================================================

// Single (cell, local-count) record exchanged across ranks during the shuffle.
// Layout matches MPI_BYTE wire format; do not add padding.
struct CellEntry {
    std::int32_t qx;
    std::int32_t qy;
    std::int32_t qz;
    std::int32_t cnt;
};
static_assert(sizeof(CellEntry) == 16, "CellEntry must be exactly 16 bytes for MPI_BYTE wire");

// Hash for the local cell map.
struct CellHasher {
    std::size_t operator()(const Eigen::RowVector3i& v) const {
        std::uint64_t h = (std::uint64_t)v[0] * 73856093u
                        ^ (std::uint64_t)v[1] * 19349663u
                        ^ (std::uint64_t)v[2] * 83492791u;
        return static_cast<std::size_t>(h);
    }
};

// Compute owner rank for a cell. Four strategies:
//
//   "samplesort"    DEFAULT. Each rank samples K Morton codes from its local
//                    cells; samples are MPI_Allgather'd, sorted, and (N-1)
//                    splitters are picked at uniform sample indices. Each
//                    rank's owned range is the inter-splitter interval. This
//                    gives near-perfect load balance regardless of cell
//                    distribution AND preserves SFC locality within rank.
//                    Avoids the ~10GB-per-rank send/recv blow-up that naive
//                    Morton-range partition causes on skewed/clustered data.
//
//   "morton-range"  Partition the actual observed Morton-code range [min, max]
//                    into N equal sub-ranges. Cheap (no sample collection)
//                    but assumes Morton density is uniform; on clustered
//                    inputs (HACC particles in a corner of the bbox), one rank
//                    gets all the cells and MPI_Alltoallv displacements
//                    overflow int. Kept for ablation.
//
//   "hash"          Uniform balance, no SFC locality. Use as a fallback when
//                    samplesort itself misbehaves. Significantly hurts
//                    encoded ratio.
//
//   "morton"        Naive fixed-bit Morton-prefix. Bad on every dataset.
//                    Kept for ablation only.
//
// Selectable via env var XNYZIP_MR_PARTITION = "samplesort" | "morton-range" |
// "hash" | "morton". Default is samplesort.
enum class PartitionMode { SAMPLESORT, MORTON_RANGE, HASH, MORTON };

inline PartitionMode pick_partition_mode() {
    if (const char* e = std::getenv("XNYZIP_MR_PARTITION")) {
        std::string s = e;
        if (s == "samplesort")   return PartitionMode::SAMPLESORT;
        if (s == "morton-range") return PartitionMode::MORTON_RANGE;
        if (s == "hash")         return PartitionMode::HASH;
        if (s == "morton")       return PartitionMode::MORTON;
    }
    return PartitionMode::SAMPLESORT;
}

inline int owner_rank_for_cell(const Eigen::RowVector3i& cell, int nranks,
                               int morton_bits,
                               std::uint64_t morton_min, std::uint64_t morton_max,
                               const std::vector<std::uint64_t>& splitters,
                               PartitionMode mode) {
    if (mode == PartitionMode::HASH) {
        std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell[0])) * 73856093u
                        ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell[1])) * 19349663u
                        ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell[2])) * 83492791u;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<int>(h % static_cast<std::uint64_t>(nranks));
    }

    std::uint64_t code = XnYZip::morton_code(cell);

    if (mode == PartitionMode::SAMPLESORT) {
        // Binary search: find the smallest splitter > code. The owner is the
        // index of the bucket that contains `code`. There are N-1 splitters
        // separating N buckets, so owner = upper_bound(splitters, code) -
        // splitters.begin() ∈ [0, N).
        auto it = std::upper_bound(splitters.begin(), splitters.end(), code);
        int owner = static_cast<int>(it - splitters.begin());
        if (owner < 0) owner = 0;
        if (owner >= nranks) owner = nranks - 1;
        return owner;
    }

    if (mode == PartitionMode::MORTON_RANGE) {
        if (morton_max <= morton_min) return 0;
        std::uint64_t range = morton_max - morton_min + 1;
        std::uint64_t off = code > morton_min ? code - morton_min : 0;
        double frac = static_cast<double>(off) / static_cast<double>(range);
        int owner = static_cast<int>(frac * static_cast<double>(nranks));
        if (owner < 0) owner = 0;
        if (owner >= nranks) owner = nranks - 1;
        return owner;
    }

    // PartitionMode::MORTON (legacy)
    if (morton_bits <= 0) morton_bits = 1;
    if (morton_bits > 63) morton_bits = 63;
    std::uint64_t scale = (morton_bits >= 63) ? std::uint64_t{1} << 63
                                              : std::uint64_t{1} << morton_bits;
    double frac = static_cast<double>(code) / static_cast<double>(scale);
    int owner = static_cast<int>(frac * static_cast<double>(nranks));
    if (owner < 0) owner = 0;
    if (owner >= nranks) owner = nranks - 1;
    return owner;
}

// Samplesort splitter selection: gather samples from all ranks, sort, pick
// (N-1) splitters at uniform indices. Returns the same vector on every rank
// (deterministic).
inline std::vector<std::uint64_t> compute_samplesort_splitters(
    const std::unordered_map<Eigen::RowVector3i, std::int64_t,
                             struct CellHasher>& local_table,
    int rank, int nranks, std::size_t samples_per_rank = 1024)
{
    // 1. Sample local_table on this rank.
    std::vector<std::uint64_t> local_samples;
    local_samples.reserve(std::min(samples_per_rank, local_table.size()));
    if (local_table.size() <= samples_per_rank) {
        for (auto& kv : local_table) {
            local_samples.push_back(XnYZip::morton_code(kv.first));
        }
    } else {
        // Reservoir sampling (Algorithm R).
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (std::uint64_t)rank);
        local_samples.resize(samples_per_rank);
        std::size_t i = 0;
        for (auto& kv : local_table) {
            std::uint64_t code = XnYZip::morton_code(kv.first);
            if (i < samples_per_rank) {
                local_samples[i] = code;
            } else {
                std::size_t j = std::uniform_int_distribution<std::size_t>(0, i)(rng);
                if (j < samples_per_rank) local_samples[j] = code;
            }
            ++i;
        }
    }
    int local_n = static_cast<int>(local_samples.size());

    // 2. Allgather counts, then bytes.
    std::vector<int> counts(nranks), displs(nranks);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < nranks; ++r) {
        displs[r] = total;
        total += counts[r];
    }
    std::vector<std::uint64_t> all_samples(total);
    MPI_Allgatherv(local_samples.data(), local_n, MPI_UINT64_T,
                   all_samples.data(), counts.data(), displs.data(), MPI_UINT64_T,
                   MPI_COMM_WORLD);

    // 3. Sort and pick splitters.
    std::sort(all_samples.begin(), all_samples.end());
    std::vector<std::uint64_t> splitters;
    if (total <= 0 || nranks <= 1) return splitters;
    splitters.reserve(nranks - 1);
    for (int i = 1; i < nranks; ++i) {
        std::size_t idx = (static_cast<std::size_t>(total) * i) / nranks;
        if (idx >= all_samples.size()) idx = all_samples.size() - 1;
        splitters.push_back(all_samples[idx]);
    }
    if (rank == 0) {
        std::printf("[mapreduce] samplesort: gathered %d samples, "
                    "%zu splitters [first=%llu, last=%llu]\n",
                    total, splitters.size(),
                    splitters.empty() ? 0ULL : (unsigned long long)splitters.front(),
                    splitters.empty() ? 0ULL : (unsigned long long)splitters.back());
    }
    return splitters;
}

// Quantize a chunk of points using the global offset (so all ranks share the
// same coordinate frame). Returns the per-cell local counts merged into
// `local_table`.
inline void map_quantize_into_local_table(
    const std::vector<Eigen::RowVector3f>& chunk_pts,
    const Eigen::RowVector3f& global_offset,
    const Args& a,
    std::unordered_map<Eigen::RowVector3i, std::int64_t, CellHasher>& local_table)
{
    using namespace XnYZip;
    QUANTIZER_TYPE qt = (a.quantizer == "cube") ? QUANTIZER_TYPE::CUBE
                                                : QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
    std::unique_ptr<BaseQuantizer<float>> quantizer;
    if (qt == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
        quantizer = std::make_unique<TruncatedOctahedronQuantizer<float>>(a.bound);
    } else {
        quantizer = std::make_unique<CubeQuantizer<float>>(a.bound);
    }

    // Apply the global offset (subtract). shift_points<float>() recomputes
    // offset from local min; we must not use it here.
    std::vector<Eigen::RowVector3f> shifted(chunk_pts.size());
    for (std::size_t i = 0; i < chunk_pts.size(); ++i) {
        shifted[i] = chunk_pts[i] - global_offset;
    }

    std::vector<float> params;
    auto qpts = quantizer->quantize(shifted, params);
    // For TO and cube quantizers, params is empty. Validate to avoid silent breakage:
    if (!params.empty()) {
        throw std::runtime_error("mapreduce mode requires stateless quantizer (got non-empty params)");
    }
    for (const auto& q : qpts) {
        local_table[q] += 1;
    }
}

// Run the full mapreduce pipeline on this rank. Returns 0 on success.
// Writes a single part file + manifest entry for this rank.
int run_mapreduce(const Args& args, MPI_File fh,
                  std::uint64_t aligned_size,
                  std::uint64_t num_blocks,
                  int rank, int nranks,
                  std::ofstream& part, std::ofstream& meta,
                  // Output stats (filled in for rank's own row):
                  std::uint64_t& out_local_input_bytes,
                  std::uint64_t& out_local_compressed_bytes,
                  double& out_t_read,
                  double& out_t_quantize,
                  double& out_t_shuffle,
                  double& out_t_encode,
                  double& out_t_write)
{
    using namespace XnYZip;

    // ---- Phase A: 1st pass over my chunks → compute local min/max for global bbox ----
    double tA = MPI_Wtime();
    Eigen::RowVector3f local_min(std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max());
    Eigen::RowVector3f local_max(std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::lowest());
    std::uint64_t my_total_input_bytes = 0;
    std::uint64_t my_block_count = 0;
    for (std::uint64_t b = static_cast<std::uint64_t>(rank); b < num_blocks;
         b += static_cast<std::uint64_t>(nranks)) {
        std::uint64_t offset = b * args.block_size;
        std::uint64_t nbytes = std::min<std::uint64_t>(args.block_size, aligned_size - offset);
        std::vector<char> in(nbytes);
        MPI_File_read_at(fh, static_cast<MPI_Offset>(offset), in.data(),
                         static_cast<int>(nbytes), MPI_BYTE, MPI_STATUS_IGNORE);
        std::uint64_t n = nbytes / POINT_BYTES;
        const float* f = reinterpret_cast<const float*>(in.data());
        for (std::uint64_t i = 0; i < n; ++i) {
            float x = f[3 * i + 0], y = f[3 * i + 1], z = f[3 * i + 2];
            if (x < local_min[0]) local_min[0] = x;  if (x > local_max[0]) local_max[0] = x;
            if (y < local_min[1]) local_min[1] = y;  if (y > local_max[1]) local_max[1] = y;
            if (z < local_min[2]) local_min[2] = z;  if (z > local_max[2]) local_max[2] = z;
        }
        my_total_input_bytes += nbytes;
        my_block_count++;
    }
    double tB = MPI_Wtime();
    if (rank == 0) {
        std::printf("[mapreduce] phase A (bbox scan) done in %.2fs\n", tB - tA);
    }

    // Reduce to global bbox
    Eigen::RowVector3f global_min, global_max;
    MPI_Allreduce(local_min.data(), global_min.data(), 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max.data(), global_max.data(), 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    Eigen::RowVector3f global_offset = global_min;  // shift_points subtracts min

    if (rank == 0) {
        std::printf("[mapreduce] global bbox: min=(%g,%g,%g) max=(%g,%g,%g)\n",
                    global_min[0], global_min[1], global_min[2],
                    global_max[0], global_max[1], global_max[2]);
    }

    // ---- Phase B (Map): 2nd pass → quantize each chunk with global offset, ----
    // ---- accumulate into local hash table.                                  ----
    std::unordered_map<Eigen::RowVector3i, std::int64_t, CellHasher> local_table;
    local_table.reserve(my_total_input_bytes / POINT_BYTES / 4);  // estimate

    double tC = MPI_Wtime();
    for (std::uint64_t b = static_cast<std::uint64_t>(rank); b < num_blocks;
         b += static_cast<std::uint64_t>(nranks)) {
        std::uint64_t offset = b * args.block_size;
        std::uint64_t nbytes = std::min<std::uint64_t>(args.block_size, aligned_size - offset);
        std::vector<char> in(nbytes);
        MPI_File_read_at(fh, static_cast<MPI_Offset>(offset), in.data(),
                         static_cast<int>(nbytes), MPI_BYTE, MPI_STATUS_IGNORE);
        auto pts = bytes_to_points(in);
        try {
            map_quantize_into_local_table(pts, global_offset, args, local_table);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d block %llu: map phase: %s\n",
                         rank, (unsigned long long)b, e.what());
            MPI_Abort(MPI_COMM_WORLD, 8);
        }
    }
    double tD = MPI_Wtime();
    out_t_read = (tB - tA);                      // 1st pass (bbox scan only)
    out_t_quantize = (tD - tC) - (tB - tA);      // 2nd pass minus its own re-read time (approx; included in tD-tC)
    if (rank == 0) {
        std::printf("[mapreduce] phase B (map: quantize+local dedup) done in %.2fs, "
                    "rank0 local_table=%zu cells\n", tD - tC, local_table.size());
    }

    // ---- Compute global max quantized coord + Morton min/max for partition ----
    int local_max_q_pre = 0;
    std::uint64_t local_morton_min = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t local_morton_max = 0;
    for (auto& kv : local_table) {
        const Eigen::RowVector3i& c = kv.first;
        local_max_q_pre = std::max({local_max_q_pre, c[0], c[1], c[2]});
        std::uint64_t code = XnYZip::morton_code(c);
        if (code < local_morton_min) local_morton_min = code;
        if (code > local_morton_max) local_morton_max = code;
    }
    if (local_table.empty()) {  // sentinel for empty rank
        local_morton_min = 0;
        local_morton_max = 0;
    }
    int global_max_q_pre = 0;
    MPI_Allreduce(&local_max_q_pre, &global_max_q_pre, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::uint64_t global_morton_min = 0, global_morton_max = 0;
    MPI_Allreduce(&local_morton_min, &global_morton_min, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_morton_max, &global_morton_max, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

    int axis_bits = 1;
    {
        int t = global_max_q_pre;
        axis_bits = 1;
        while (t > 1) { axis_bits++; t >>= 1; }
        if (axis_bits < 1) axis_bits = 1;
    }
    int morton_bits = std::min(63, 3 * axis_bits);
    if (rank == 0) {
        std::printf("[mapreduce] global max q = %d, axis_bits=%d, morton_bits=%d\n",
                    global_max_q_pre, axis_bits, morton_bits);
        std::printf("[mapreduce] global morton range = [%llu, %llu]\n",
                    (unsigned long long)global_morton_min,
                    (unsigned long long)global_morton_max);
    }

    // ---- Phase C (Shuffle): MPI_Alltoallv by partition function ----
    PartitionMode part_mode = pick_partition_mode();
    if (rank == 0) {
        const char* name = part_mode == PartitionMode::SAMPLESORT   ? "samplesort"
                         : part_mode == PartitionMode::HASH         ? "hash"
                         : part_mode == PartitionMode::MORTON_RANGE ? "morton-range"
                                                                    : "morton";
        std::printf("[mapreduce] partition mode = %s\n", name);
    }

    // Samplesort needs an extra Allgather of K samples per rank to compute
    // splitters before we can call owner_rank_for_cell. For other modes the
    // splitters vector stays empty and is unused.
    std::vector<std::uint64_t> splitters;
    if (part_mode == PartitionMode::SAMPLESORT) {
        double tS = MPI_Wtime();
        splitters = compute_samplesort_splitters(local_table, rank, nranks, /*samples_per_rank=*/1024);
        if (rank == 0) {
            std::printf("[mapreduce] samplesort splitter selection: %.2fs\n", MPI_Wtime() - tS);
        }
    }

    std::vector<std::vector<CellEntry>> send_buckets(nranks);
    for (auto& kv : local_table) {
        const Eigen::RowVector3i& c = kv.first;
        int owner = owner_rank_for_cell(c, nranks, morton_bits,
                                        global_morton_min, global_morton_max,
                                        splitters, part_mode);
        std::int32_t cnt32 = static_cast<std::int32_t>(std::min<std::int64_t>(kv.second, INT32_MAX));
        send_buckets[owner].push_back(CellEntry{c[0], c[1], c[2], cnt32});
    }
    local_table.clear();  // free memory

    // Pack contiguous send buffer + sendcounts/senddispls
    std::vector<int> sendcounts(nranks, 0), senddispls(nranks, 0);
    std::vector<int> recvcounts(nranks, 0), recvdispls(nranks, 0);
    std::size_t total_send = 0;
    for (int r = 0; r < nranks; ++r) {
        sendcounts[r] = static_cast<int>(send_buckets[r].size() * sizeof(CellEntry));
        senddispls[r] = static_cast<int>(total_send);
        total_send += send_buckets[r].size() * sizeof(CellEntry);
    }
    std::vector<char> sendbuf(total_send);
    for (int r = 0; r < nranks; ++r) {
        if (send_buckets[r].empty()) continue;
        std::memcpy(sendbuf.data() + senddispls[r],
                    send_buckets[r].data(),
                    send_buckets[r].size() * sizeof(CellEntry));
    }
    // Free per-bucket vectors after copy
    std::vector<std::vector<CellEntry>>().swap(send_buckets);

    // Exchange sizes
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                 recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::size_t total_recv = 0;
    for (int r = 0; r < nranks; ++r) {
        recvdispls[r] = static_cast<int>(total_recv);
        total_recv += recvcounts[r];
    }
    std::vector<char> recvbuf(total_recv);

    double tE = MPI_Wtime();
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), senddispls.data(), MPI_BYTE,
                  recvbuf.data(), recvcounts.data(), recvdispls.data(), MPI_BYTE,
                  MPI_COMM_WORLD);
    double tF = MPI_Wtime();
    out_t_shuffle = tF - tE;
    std::vector<char>().swap(sendbuf);  // free
    if (rank == 0) {
        std::printf("[mapreduce] phase C (alltoallv shuffle) done in %.2fs, "
                    "rank0 received %zu bytes\n", tF - tE, total_recv);
    }

    // ---- Phase D (Reduce): aggregate received cells in MY owned table ----
    std::unordered_map<Eigen::RowVector3i, std::int64_t, CellHasher> owned_table;
    {
        const CellEntry* buf = reinterpret_cast<const CellEntry*>(recvbuf.data());
        std::size_t n = total_recv / sizeof(CellEntry);
        owned_table.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            Eigen::RowVector3i k(buf[i].qx, buf[i].qy, buf[i].qz);
            owned_table[k] += static_cast<std::int64_t>(buf[i].cnt);
        }
    }
    std::vector<char>().swap(recvbuf);

    // Sanity print
    std::printf("[mapreduce] rank %d: owned_table size = %zu cells\n", rank, owned_table.size());

    // ---- Phase E (Encode): use BlockCompressorRLE::encode_cells_with_counts ----
    std::vector<Eigen::RowVector3i> cells;
    std::vector<int> counts;
    cells.reserve(owned_table.size());
    counts.reserve(owned_table.size());
    for (auto& kv : owned_table) {
        cells.push_back(kv.first);
        counts.push_back(static_cast<int>(std::min<std::int64_t>(kv.second, INT32_MAX)));
    }
    owned_table.clear();

    // Use global max quantized coord computed earlier (Phase C prep).
    // Single-axis range across x,y,z for simplicity (monolithic compress()
    // computes per-axis but that requires additional reductions; single-range
    // costs only a few extra bits in bitpacker_blk encoding).
    int range_x = global_max_q_pre, range_y = global_max_q_pre, range_z = global_max_q_pre;

    bool is_hilbert = (args.curve == "-h");
    XnYZip::QUANTIZER_TYPE qt = (args.quantizer == "cube") ? XnYZip::QUANTIZER_TYPE::CUBE
                                                           : XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
    BlockCompressorRLE<float> rle_enc(qt, args.bound, false, args.direct_threshold);

    double tG = MPI_Wtime();
    std::vector<std::uint8_t> compressed;
    try {
        compressed = rle_enc.encode_cells_with_counts(
            cells, counts,
            range_x, range_y, range_z,
            global_offset, is_hilbert);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d encode: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 9);
    }
    double tH = MPI_Wtime();
    out_t_encode = tH - tG;

    // ---- Phase F (Write): single part file per rank ----
    std::uint64_t out_offset = static_cast<std::uint64_t>(part.tellp());
    part.write(reinterpret_cast<const char*>(compressed.data()),
               static_cast<std::streamsize>(compressed.size()));
    part.flush();
    double tI = MPI_Wtime();
    out_t_write = tI - tH;

    meta << "{"
         << "\"rank\": " << rank
         << ", \"mode\": \"mapreduce\""
         << ", \"input_bytes\": " << my_total_input_bytes
         << ", \"input_blocks_owned\": " << my_block_count
         << ", \"owned_cells\": " << cells.size()
         << ", \"part_offset\": " << out_offset
         << ", \"compressed_size\": " << compressed.size()
         << "}\n";
    meta.flush();

    out_local_input_bytes = my_total_input_bytes;
    out_local_compressed_bytes = compressed.size();

    std::printf("[mapreduce] rank %d: input=%llu B, encoded %zu cells → %zu B "
                "(read=%.2fs map=%.2fs shuffle=%.2fs encode=%.2fs write=%.2fs)\n",
                rank, (unsigned long long)my_total_input_bytes,
                cells.size(), compressed.size(),
                out_t_read, out_t_quantize, out_t_shuffle, out_t_encode, out_t_write);

    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    // Force line-buffered stdout/stderr so progress lines appear in PBS log
    // immediately rather than being held until the job exits.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        if (rank == 0) std::fprintf(stderr, "arg error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- discover file size on rank 0, broadcast ----
    std::uint64_t input_size = 0;
    if (rank == 0) {
        try {
            input_size = XnYZip::get_file_size(args.input);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank 0: %s\n", e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&input_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    std::uint64_t aligned_size = input_size - (input_size % POINT_BYTES);
    std::uint64_t num_blocks = (aligned_size + args.block_size - 1) / args.block_size;
    if (num_blocks == 0) {
        if (rank == 0) std::fprintf(stderr, "input too small (%llu bytes)\n",
                                    (unsigned long long)input_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("input=%s size=%llu aligned=%llu block_size=%llu num_blocks=%llu nranks=%d\n",
                    args.input.c_str(),
                    (unsigned long long)input_size,
                    (unsigned long long)aligned_size,
                    (unsigned long long)args.block_size,
                    (unsigned long long)num_blocks,
                    nranks);
    }

    // ---- open shared input ----
    MPI_File fh = MPI_FILE_NULL;
    int rc = MPI_File_open(MPI_COMM_WORLD, args.input.c_str(),
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) std::fprintf(stderr, "MPI_File_open failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- rank-local outputs ----
    std::ostringstream rank_str;
    rank_str.width(4); rank_str.fill('0'); rank_str << rank;
    std::string part_path = args.output_dir + "/part_rank" + rank_str.str() + ".bin";
    std::string meta_path = args.output_dir + "/manifest_rank" + rank_str.str() + ".jsonl";
    std::ofstream part(part_path, std::ios::binary);
    std::ofstream meta(meta_path);
    if (!part.is_open() || !meta.is_open()) {
        std::fprintf(stderr, "rank %d: cannot open output files in %s\n", rank, args.output_dir.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- shared-codebook warmup: rank 0 fits codebooks on block 0 and ----
    // ---- broadcasts the serialized profile to all ranks                 ----
    XnYZip::BlockEncodingProfile shared_profile;
    double t_warmup = 0.0;
    if (args.shared_codebook) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_warmup_start = MPI_Wtime();

        std::vector<std::uint8_t> profile_blob;
        std::uint64_t warmup_offset = 0;
        std::uint64_t warmup_nbytes = std::min<std::uint64_t>(args.block_size, aligned_size);

        if (rank == 0) {
            std::vector<char> warmup_in(warmup_nbytes);
            MPI_File_read_at(fh, static_cast<MPI_Offset>(warmup_offset), warmup_in.data(),
                             static_cast<int>(warmup_nbytes), MPI_BYTE, MPI_STATUS_IGNORE);
            auto warmup_pts = bytes_to_points(warmup_in);
            try {
                XnYZip::QUANTIZER_TYPE qt =
                    (args.quantizer == "cube") ? XnYZip::QUANTIZER_TYPE::CUBE
                                               : XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
                bool is_hilbert = (args.curve == "-h");
                XnYZip::BlockCompressor<float> c(qt, args.bound, false, args.direct_threshold);
                auto [warmup_bytes, prof] = c.fit_profile(warmup_pts, is_hilbert);
                shared_profile = std::move(prof);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "rank 0 warmup: %s\n", e.what());
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            profile_blob = XnYZip::serialize_profile(shared_profile);
            std::printf("[shared-codebook] rank 0 fit profile: blob=%zu bytes "
                        "(repos_meta=%zu, quads_meta=%zu)\n",
                        profile_blob.size(),
                        shared_profile.repos_huff_meta.size(),
                        shared_profile.quads_huff_meta.size());
        }

        // Broadcast blob size, then bytes.
        std::uint64_t blob_size = profile_blob.size();
        MPI_Bcast(&blob_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        if (rank != 0) profile_blob.resize(blob_size);
        MPI_Bcast(profile_blob.data(), static_cast<int>(blob_size), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) shared_profile = XnYZip::deserialize_profile(profile_blob);

        if (!shared_profile.valid()) {
            if (rank == 0) std::fprintf(stderr, "[shared-codebook] invalid profile after broadcast\n");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t_warmup = MPI_Wtime() - t_warmup_start;
        if (rank == 0) {
            std::printf("[shared-codebook] warmup + broadcast: %.4f s\n", t_warmup);
        }
    }

    const XnYZip::BlockEncodingProfile* profile_ptr =
        args.shared_codebook ? &shared_profile : nullptr;

    // ---- per-rank accumulators ----
    double t_read = 0.0, t_comp = 0.0, t_write = 0.0;
    std::uint64_t local_input_bytes = 0;
    std::uint64_t local_compressed_bytes = 0;
    std::uint64_t local_blocks = 0;

    // How many blocks does THIS rank own (used for progress display)?
    std::uint64_t my_total_blocks = 0;
    for (std::uint64_t b = static_cast<std::uint64_t>(rank); b < num_blocks;
         b += static_cast<std::uint64_t>(nranks)) {
        my_total_blocks++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        std::printf("[progress] starting compression: %llu blocks across %d ranks "
                    "(rank 0 will process %llu blocks)\n",
                    (unsigned long long)num_blocks, nranks,
                    (unsigned long long)my_total_blocks);
    }

    // ---- mapreduce mode: replace per-chunk loop entirely ----
    if (args.mode == "mapreduce") {
        double mr_read = 0, mr_quant = 0, mr_shuffle = 0, mr_encode = 0, mr_write = 0;
        run_mapreduce(args, fh, aligned_size, num_blocks, rank, nranks,
                      part, meta,
                      local_input_bytes, local_compressed_bytes,
                      mr_read, mr_quant, mr_shuffle, mr_encode, mr_write);
        // Map mapreduce phase timings into the per-chunk schema for summary:
        t_read  = mr_read;
        t_comp  = mr_quant + mr_shuffle + mr_encode;
        t_write = mr_write;
        local_blocks = my_total_blocks;  // blocks "owned" (read in Phase A/B)
    } else {

    std::uint64_t my_block_idx = 0;
    for (std::uint64_t b = static_cast<std::uint64_t>(rank); b < num_blocks;
         b += static_cast<std::uint64_t>(nranks)) {
        std::uint64_t offset = b * args.block_size;
        std::uint64_t nbytes = std::min<std::uint64_t>(args.block_size, aligned_size - offset);

        std::vector<char> in(nbytes);

        double a_read = MPI_Wtime();
        MPI_File_read_at(fh, static_cast<MPI_Offset>(offset), in.data(),
                         static_cast<int>(nbytes), MPI_BYTE, MPI_STATUS_IGNORE);
        double a_decode = MPI_Wtime();

        auto pts = bytes_to_points(in);
        // Keep an unmodified copy for roundtrip verification, since compress()
        // reorders the input vector in-place via the SFC permutation.
        std::vector<Eigen::RowVector3f> original_pts;
        if (args.verify_roundtrip) original_pts = pts;
        std::vector<std::uint8_t> compressed;
        try {
            compressed = compress_block(pts, args, profile_ptr);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d block %llu: compress failed: %s\n",
                         rank, (unsigned long long)b, e.what());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        double a_comp = MPI_Wtime();

        if (args.verify_roundtrip) {
            // RLE mode decompression returns the deduplicated unique points
            // only (not the original points), so the per-point comparison
            // below would be invalid. Skip with a one-time notice.
            if (args.mode == "-rle") {
                static bool warned = false;
                if (!warned && rank == 0) {
                    std::printf("[verify] -rle mode: per-point roundtrip check skipped "
                                "(RLE deduplicates; full-point recovery requires the "
                                "external recovery pipeline). Compression itself is "
                                "self-checked by the encoder.\n");
                    warned = true;
                }
                goto skip_verify;
            }
            try {
                XnYZip::QUANTIZER_TYPE qt =
                    (args.quantizer == "cube") ? XnYZip::QUANTIZER_TYPE::CUBE
                                               : XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
                XnYZip::BlockDecompressor<float> d(qt, args.bound);
                auto recovered = d.decompress(compressed);
                if (recovered.size() != original_pts.size()) {
                    std::fprintf(stderr,
                        "rank %d block %llu: roundtrip size mismatch (orig=%zu rec=%zu)\n",
                        rank, (unsigned long long)b, original_pts.size(), recovered.size());
                    MPI_Abort(MPI_COMM_WORLD, 5);
                }
                // The compressor sorts points along the SFC, so original_pts
                // and recovered are in the SAME order (compress() permutes
                // original_pts in place; compress_with_profile() does the same).
                // We can therefore compare elementwise on the post-compress
                // permutation, which is `pts` (modified in place above).
                float max_err = 0.0f;
                std::uint64_t n_overflow = 0;
                for (std::size_t i = 0; i < pts.size(); ++i) {
                    float err = (pts[i] - recovered[i]).norm();
                    if (err > max_err) max_err = err;
                    if (err > args.bound) ++n_overflow;
                }
                if (n_overflow > 0) {
                    std::fprintf(stderr,
                        "rank %d block %llu: %llu points exceed bound (max_err=%.6f bound=%.6f)\n",
                        rank, (unsigned long long)b, (unsigned long long)n_overflow,
                        max_err, args.bound);
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }
                std::printf("[verify] rank %d block %llu: %zu points OK (max_err=%.6e <= %.6e)\n",
                            rank, (unsigned long long)b, recovered.size(), max_err, args.bound);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "rank %d block %llu: roundtrip exception: %s\n",
                             rank, (unsigned long long)b, e.what());
                MPI_Abort(MPI_COMM_WORLD, 7);
            }
            skip_verify:;
        }

        std::uint64_t out_offset = static_cast<std::uint64_t>(part.tellp());
        part.write(reinterpret_cast<const char*>(compressed.data()),
                   static_cast<std::streamsize>(compressed.size()));
        double a_write = MPI_Wtime();

        meta << "{"
             << "\"block_id\":" << b
             << ",\"input_offset\":" << offset
             << ",\"input_size\":" << nbytes
             << ",\"rank\":" << rank
             << ",\"part_offset\":" << out_offset
             << ",\"compressed_size\":" << compressed.size()
             << "}\n";

        t_read  += a_decode - a_read;
        t_comp  += a_comp   - a_decode;
        t_write += a_write  - a_comp;
        local_input_bytes      += nbytes;
        local_compressed_bytes += compressed.size();
        local_blocks++;

        // Per-block progress: every rank prints one line per finished block.
        // Format chosen so it's grep-friendly (`grep '\[progress\]'`).
        my_block_idx++;
        double dt_read  = a_decode - a_read;
        double dt_comp  = a_comp   - a_decode;
        double dt_write = a_write  - a_comp;
        double elapsed  = a_write  - t0;
        double chunk_ratio = (compressed.size() > 0)
            ? double(nbytes) / double(compressed.size()) : 0.0;
        std::printf("[progress] rank %d  block %llu  (mine %llu/%llu)  "
                    "read=%.2fs comp=%.2fs write=%.3fs  in=%.1fMB out=%.2fMB ratio=%.1f  "
                    "rank_elapsed=%.1fs\n",
                    rank,
                    (unsigned long long)b,
                    (unsigned long long)my_block_idx,
                    (unsigned long long)my_total_blocks,
                    dt_read, dt_comp, dt_write,
                    nbytes / 1e6, compressed.size() / 1e6, chunk_ratio,
                    elapsed);
    }
    } // end of else (per-chunk mode)
    part.flush();
    meta.flush();
    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_wall = t1 - t0;

    // ---- reductions ----
    std::uint64_t g_in = 0, g_out = 0, g_blocks = 0;
    MPI_Reduce(&local_input_bytes,      &g_in,     1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compressed_bytes, &g_out,    1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_blocks,           &g_blocks, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_read = 0, max_comp = 0, max_write = 0, max_wall = 0;
    double min_wall = 0;
    MPI_Reduce(&t_read,     &max_read,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp,     &max_comp,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_write,    &max_write, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_wall, &max_wall,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_wall, &min_wall,  1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // ---- per-rank stats: write to a JSONL on rank 0 via gather ----
    std::vector<std::uint64_t> all_in(nranks), all_out(nranks), all_blocks(nranks);
    std::vector<double> all_wall(nranks);
    MPI_Gather(&local_input_bytes,      1, MPI_UINT64_T, all_in.data(),     1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_compressed_bytes, 1, MPI_UINT64_T, all_out.data(),    1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_blocks,           1, MPI_UINT64_T, all_blocks.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_wall,             1, MPI_DOUBLE,   all_wall.data(),   1, MPI_DOUBLE,   0, MPI_COMM_WORLD);

    if (rank == 0) {
        double ratio = (g_out > 0) ? (double)g_in / (double)g_out : 0.0;
        double end_to_end_throughput_GBps = (max_wall > 0) ? (g_in / max_wall) / 1e9 : 0.0;
        double parallel_efficiency = (max_wall > 0 && min_wall > 0) ? (min_wall / max_wall) : 0.0;

        std::printf(
            "\n=== run summary ===\n"
            "nranks=%d num_blocks=%llu\n"
            "total_input_bytes=%llu total_compressed_bytes=%llu ratio=%.4f\n"
            "wall_max_s=%.4f wall_min_s=%.4f load_balance=%.4f\n"
            "phase_max(read=%.4f comp=%.4f write=%.4f)\n"
            "end_to_end_throughput=%.4f GB/s\n",
            nranks,
            (unsigned long long)g_blocks,
            (unsigned long long)g_in, (unsigned long long)g_out, ratio,
            max_wall, min_wall, parallel_efficiency,
            max_read, max_comp, max_write,
            end_to_end_throughput_GBps);

        std::ofstream summary(args.output_dir + "/summary.json");
        summary << "{\n"
                << "  \"nranks\": " << nranks << ",\n"
                << "  \"num_blocks\": " << g_blocks << ",\n"
                << "  \"block_size\": " << args.block_size << ",\n"
                << "  \"shared_codebook\": " << (args.shared_codebook ? "true" : "false") << ",\n"
                << "  \"warmup_s\": " << t_warmup << ",\n"
                << "  \"input_bytes\": " << g_in << ",\n"
                << "  \"compressed_bytes\": " << g_out << ",\n"
                << "  \"ratio\": " << ratio << ",\n"
                << "  \"wall_max_s\": " << max_wall << ",\n"
                << "  \"wall_min_s\": " << min_wall << ",\n"
                << "  \"phase_max\": {\"read\": " << max_read
                << ", \"comp\": " << max_comp
                << ", \"write\": " << max_write << "},\n"
                << "  \"end_to_end_throughput_GBps\": " << end_to_end_throughput_GBps << ",\n"
                << "  \"per_rank\": [\n";
        for (int r = 0; r < nranks; ++r) {
            summary << "    {\"rank\": " << r
                    << ", \"input_bytes\": " << all_in[r]
                    << ", \"compressed_bytes\": " << all_out[r]
                    << ", \"blocks\": " << all_blocks[r]
                    << ", \"wall_s\": " << all_wall[r]
                    << "}" << (r + 1 == nranks ? "" : ",") << "\n";
        }
        summary << "  ]\n}\n";
        summary.close();

        // Also write a merged manifest pointer file (we don't merge JSONL contents,
        // just record which rank-files compose the dataset). Downstream tools can
        // concat the per-rank manifests with `cat manifest_rank*.jsonl`.
        std::ofstream merged(args.output_dir + "/manifest_index.jsonl");
        for (int r = 0; r < nranks; ++r) {
            std::ostringstream rs;
            rs.width(4); rs.fill('0'); rs << r;
            merged << "{\"rank\": " << r
                   << ", \"part\": \"part_rank" << rs.str() << ".bin\""
                   << ", \"manifest\": \"manifest_rank" << rs.str() << ".jsonl\"}\n";
        }
        merged.close();
    }

    MPI_Finalize();
    return 0;
}
