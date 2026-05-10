// Distributed-RLE decompression utility.
//
// Reads all rank-local part files produced by `XnYZip_mpi --mode mapreduce`,
// decodes each into (cells, counts) via BlockDecompressorRLE::decompress_to_cells,
// expands cells via the quantizer (each cell appears `count` times), unshifts
// by the global offset, and writes the recovered point cloud to disk.
//
// Output ORDER NOTE: mapreduce mode does not preserve original input order.
// Recovered points are emitted in part-file order (i.e., by rank), with each
// rank's cells in the order the encoder used (Morton/Hilbert SFC after the
// rank's local SFC sort). For point-cloud workloads where particles are
// distinguishable only by position (HACC etc.) this is fine; for
// position-AND-index-meaningful data (mesh attributes etc.) use the per-chunk
// `-rle` mode which preserves input order within each chunk.
//
// Usage:
//   XnYZip_mr_decompress --input-dir DIR --output PATH \
//                        [--quantizer cube|to] [--bound F]
//
// `--input-dir` should contain `part_rank0000.bin`, `part_rank0001.bin`, ...
// `--bound` must match what was used at compression time.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lib.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string input_dir;
    std::string output;
    std::string quantizer = "to";
    float bound = 1e-3f;
};

Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if      (k == "--input-dir") a.input_dir = need(k.c_str());
        else if (k == "--output")    a.output    = need(k.c_str());
        else if (k == "--quantizer") a.quantizer = need(k.c_str());
        else if (k == "--bound")     a.bound     = std::stof(need(k.c_str()));
        else if (k == "-h" || k == "--help") {
            std::fprintf(stderr,
                "Usage: %s --input-dir DIR --output PATH "
                "[--quantizer cube|to] [--bound F]\n", argv[0]);
            std::exit(0);
        }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.input_dir.empty() || a.output.empty()) {
        throw std::runtime_error("--input-dir and --output are required");
    }
    return a;
}

std::vector<std::string> list_part_files(const std::string& dir) {
    std::regex pat(R"(part_rank(\d+)\.bin)");
    std::vector<std::pair<int, std::string>> paths;
    for (auto& e : fs::directory_iterator(dir)) {
        std::string name = e.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, pat)) {
            int rank = std::stoi(m[1].str());
            paths.emplace_back(rank, e.path().string());
        }
    }
    std::sort(paths.begin(), paths.end(),
              [](auto& a, auto& b){ return a.first < b.first; });
    std::vector<std::string> out;
    out.reserve(paths.size());
    for (auto& p : paths) out.push_back(p.second);
    return out;
}

std::vector<std::uint8_t> read_file_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

} // namespace

int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }

    XnYZip::QUANTIZER_TYPE qt = (args.quantizer == "cube")
        ? XnYZip::QUANTIZER_TYPE::CUBE
        : XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON;
    XnYZip::BlockDecompressorRLE<float> dec(qt, args.bound);

    auto parts = list_part_files(args.input_dir);
    if (parts.empty()) {
        std::fprintf(stderr, "no part_rank*.bin files in %s\n", args.input_dir.c_str());
        return 2;
    }
    std::printf("decompressing %zu part files from %s\n", parts.size(), args.input_dir.c_str());

    std::ofstream out(args.output, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot open output %s\n", args.output.c_str());
        return 3;
    }

    std::uint64_t total_cells = 0;
    std::uint64_t total_points = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        auto buf = read_file_bin(parts[i]);
        if (buf.empty()) {
            std::printf("  [%zu] %s : empty (rank had no cells), skipping\n",
                        i, parts[i].c_str());
            continue;
        }
        auto table = dec.decompress_to_cells(buf);
        // Sanity: zero-cell parts (e.g. rank that ended up owning no cells in
        // mapreduce) still emit a tiny header. Tolerate them.
        if (table.cells.empty()) {
            std::printf("  [%zu] %s : 0 cells, skipping\n", i, parts[i].c_str());
            continue;
        }
        auto pts = dec.expand_cells_to_points(table);
        total_cells += table.cells.size();
        total_points += pts.size();
        for (const auto& p : pts) {
            float xyz[3] = {p[0], p[1], p[2]};
            out.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
        }
        std::printf("  [%zu] %s : %zu cells → %zu points\n",
                    i, parts[i].c_str(), table.cells.size(), pts.size());
    }

    out.close();
    std::printf("\nDone. Total: %llu cells → %llu points → %llu bytes\n",
                (unsigned long long)total_cells,
                (unsigned long long)total_points,
                (unsigned long long)total_points * 12);
    return 0;
}
