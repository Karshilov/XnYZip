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
#include <sstream>
#include <stdexcept>
#include <string>
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
    std::string mode = "-normal";   // "-normal" | "-rle"
    float bound = 1e-3f;
    int direct_threshold = 1024;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --input PATH --output-dir DIR --block-size BYTES --bound F "
        "[--quantizer cube|to] [--curve -z|-h] [--mode -normal|-rle] "
        "[--direct-threshold N]\n", prog);
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
                                         const Args& a) {
    using namespace XnYZip;
    bool is_hilbert = (a.curve == "-h");
    bool use_rle = (a.mode == "-rle");
    QUANTIZER_TYPE qt = (a.quantizer == "cube") ? QUANTIZER_TYPE::CUBE
                                                : QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
    if (use_rle) {
        BlockCompressorRLE<float> c(qt, a.bound, false, a.direct_threshold);
        return c.compress(pts, is_hilbert);
    }
    BlockCompressor<float> c(qt, a.bound, false, a.direct_threshold);
    return c.compress(pts, is_hilbert);
}

} // namespace

int main(int argc, char* argv[]) {
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

    // ---- per-rank accumulators ----
    double t_read = 0.0, t_comp = 0.0, t_write = 0.0;
    std::uint64_t local_input_bytes = 0;
    std::uint64_t local_compressed_bytes = 0;
    std::uint64_t local_blocks = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

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
        std::vector<std::uint8_t> compressed;
        try {
            compressed = compress_block(pts, args);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d block %llu: compress failed: %s\n",
                         rank, (unsigned long long)b, e.what());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        double a_comp = MPI_Wtime();

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
    }

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
