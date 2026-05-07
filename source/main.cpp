#include <string>
#include <zstd.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include "lib.hpp"

namespace {
struct ChunkSpec {
  bool active = false;
  std::uint64_t offset_bytes = 0;
  std::uint64_t count_bytes = 0;
};

auto parse_chunk_flags(int argc, char* argv[]) -> ChunkSpec {
  ChunkSpec spec;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto strip_eq = [&](const std::string& key) -> bool {
      if (a.rfind(key + "=", 0) == 0) {
        a = a.substr(key.size() + 1);
        return true;
      }
      return false;
    };
    if (strip_eq("--offset")) {
      spec.offset_bytes = std::stoull(a);
      spec.active = true;
    } else if (strip_eq("--count")) {
      spec.count_bytes = std::stoull(a);
      spec.active = true;
    }
  }
  return spec;
}

auto strip_chunk_flags(int argc, char* argv[]) -> std::vector<char*> {
  std::vector<char*> out;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--offset=", 0) == 0 || a.rfind("--count=", 0) == 0) continue;
    out.push_back(argv[i]);
  }
  return out;
}
}

auto main(int argc, char* argv[]) -> int
{
  ChunkSpec chunk = parse_chunk_flags(argc, argv);
  auto positional = strip_chunk_flags(argc, argv);
  int pargc = static_cast<int>(positional.size());
  char** pargv = positional.data();

  if (pargc < 7 || pargc > 8) {
    printf("Usage: %s <input_file> <quantizer_type (cube/to)> <L2 bound> <curve_type (-z/-h)> <-rle/-normal> <decompression_file_name> [direct_threshold (default 1024, 0=force block)] [--offset=BYTES] [--count=BYTES]\n", argv[0]);
    return 1;
  }

  auto const file_name = pargv[1];
  std::vector<Eigen::RowVector3f> points;
  if (chunk.active) {
    auto file_size = XnYZip::get_file_size(file_name);
    points = XnYZip::read_file_chunk<float>(file_name, chunk.offset_bytes, chunk.count_bytes);
    std::cout << "Chunk mode: file_size=" << file_size
              << " offset=" << chunk.offset_bytes
              << " count=" << chunk.count_bytes
              << " points=" << points.size() << std::endl;
  } else {
    points = XnYZip::read_file<float>(file_name);
  }
  std::vector<Eigen::RowVector3f> recovered_points;

  bool is_hilbert = std::string(pargv[4]) == "-h";
  bool use_rle = std::string(pargv[5]) == "-rle";
  int direct_threshold = (pargc >= 8) ? std::stoi(pargv[7]) : 1024;
  std::cout << "SFC type: " << (is_hilbert ? "Hilbert" : "Z-order") << std::endl;
  std::cout << "Use RLE: " << (use_rle ? "Yes" : "No") << std::endl;
  std::cout << "Direct threshold: " << direct_threshold << std::endl;

  uint64_t input_bytes = static_cast<uint64_t>(points.size()) * 3 * sizeof(float);
  uint64_t compressed_bytes = 0;

  if (std::string(pargv[2]) == "cube") {
    XnYZip::Timer timer;
    if (use_rle) {
      timer.start();
      auto compressor = XnYZip::BlockCompressorRLE<float>(XnYZip::QUANTIZER_TYPE::CUBE, std::stof(pargv[3]), false, direct_threshold);
      auto compressed_points = compressor.compress(points, is_hilbert);

      timer.stop("XnYZip compression");

      compressed_bytes = compressed_points.size();
      printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

      timer.start();
      auto decompressor = XnYZip::BlockDecompressorRLE<float>(XnYZip::QUANTIZER_TYPE::CUBE, std::stof(pargv[3]));
      recovered_points = decompressor.decompress(compressed_points);

      timer.stop("XnYZip decompression");
    } else {
      timer.start();
      auto compressor = XnYZip::BlockCompressor<float>(XnYZip::QUANTIZER_TYPE::CUBE, std::stof(pargv[3]), false, direct_threshold);
      auto compressed_points = compressor.compress(points, is_hilbert);

      timer.stop("XnYZip compression");

      compressed_bytes = compressed_points.size();
      printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

      timer.start();
      auto decompressor = XnYZip::BlockDecompressor<float>(XnYZip::QUANTIZER_TYPE::CUBE, std::stof(pargv[3]));
      recovered_points = decompressor.decompress(compressed_points);

      timer.stop("XnYZip decompression");
    }

  } else if (std::string(pargv[2]) == "to") {
    XnYZip::Timer timer;
    if (use_rle) {
      timer.start();
      auto compressor = XnYZip::BlockCompressorRLE<float>(XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(pargv[3]), false, direct_threshold);
      auto compressed_points = compressor.compress(points, is_hilbert);

      timer.stop("XnYZip compression");

      compressed_bytes = compressed_points.size();
      printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

      timer.start();
      auto decompressor = XnYZip::BlockDecompressorRLE<float>(XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(pargv[3]));
      recovered_points = decompressor.decompress(compressed_points);

      timer.stop("XnYZip decompression");

      printf("decompressed finished, size: %lu\n", recovered_points.size());
    } else {
      timer.start();
      auto compressor = XnYZip::BlockCompressor<float>(XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(pargv[3]), false, direct_threshold);
      auto compressed_points = compressor.compress(points, is_hilbert);

      timer.stop("XnYZip compression");

      compressed_bytes = compressed_points.size();
      printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

      timer.start();
      auto decompressor = XnYZip::BlockDecompressor<float>(XnYZip::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(pargv[3]));
      recovered_points = decompressor.decompress(compressed_points);

      timer.stop("XnYZip decompression");

      printf("decompressed finished, size: %lu\n", recovered_points.size());
    }
  }

  printf("chunk input bytes: %lu\n", input_bytes);
  printf("chunk compressed bytes: %lu\n", compressed_bytes);

  XnYZip::write_file(pargv[6], recovered_points);
  if (!use_rle) {
    double acc_mse = 0;
    float max_l2_error = 0;
    int overflow_count = 0;
    std::vector<float> errors {};
    for (int i = 0; i < points.size(); i++) {
      acc_mse += (points[i].x() - recovered_points[i].x()) * (points[i].x() - recovered_points[i].x());
      acc_mse += (points[i].y() - recovered_points[i].y()) * (points[i].y() - recovered_points[i].y());
      acc_mse += (points[i].z() - recovered_points[i].z()) * (points[i].z() - recovered_points[i].z());
      errors.push_back((points[i] - recovered_points[i]).norm());
      if ((points[i] - recovered_points[i]).norm() > std::stof(pargv[3])) {
        overflow_count++;
      }
      max_l2_error = std::max(max_l2_error, (points[i] - recovered_points[i]).norm());
    }

    XnYZip::write_file_bin("errors-" + std::string(pargv[2]) + ".bin", errors);

    printf("overflow count: %d\n", overflow_count);
    printf("Max L2 error: %f\n", max_l2_error);
    acc_mse /= points.size();
    float max_x = -1e9;
    float max_y = -1e9;
    float max_z = -1e9;
    float min_x = 1e9;
    float min_y = 1e9;
    float min_z = 1e9;
    for (auto & point : points) {
      max_x = std::max(max_x, point.x());
      max_y = std::max(max_y, point.y());
      max_z = std::max(max_z, point.z());
      min_x = std::min(min_x, point.x());
      min_y = std::min(min_y, point.y());
      min_z = std::min(min_z, point.z());
    }
    double range = (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) + (max_z - min_z) * (max_z - min_z);
    double psnr_p2p = 10.0 * std::log10(range / acc_mse);
    printf("p2p psnr: %lf\n", psnr_p2p);
    printf("MSE: %f, range: %f\n", acc_mse, range);
    printf("Max diff: %f, %f, %f\n", max_x, max_y, max_z);
  }
  return 0;
}
