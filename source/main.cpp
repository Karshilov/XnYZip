#include <string>
#include <zstd.h>
#include <cstring>
#include <vector>
#include "io/fileUtils.hpp"
#include "lib.hpp"

auto main(int argc, char* argv[]) -> int
{
  auto const file_name = argv[1];
  auto points = TonSZ::read_file<float>(file_name);
  std::vector<Eigen::RowVector3f> recovered_points;

  if (argc != 5) {
    printf("Usage: %s <input_file> <quantizer_type (cube/octa)> <L2 bound> <decompression_file_name>\n", argv[0]);
    return 1;
  }

  if (std::string(argv[2]) == "cube") {
    auto compressor = TonSZ::BlockCompressor<float>(TonSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points);

    printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

    auto decompressor = TonSZ::BlockDecompressor<float>(TonSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    recovered_points = decompressor.decompress(compressed_points);

  } else if (std::string(argv[2]) == "octa") {
    auto compressor = TonSZ::BlockCompressor<float>(TonSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points);

    printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

    auto decompressor = TonSZ::BlockDecompressor<float>(TonSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    recovered_points = decompressor.decompress(compressed_points);

    printf("decompressed finished, size: %lu\n", recovered_points.size());

  } else if (std::string(argv[2]) == "adaptive") {
    auto compressor = TonSZ::BlockCompressor<float>(TonSZ::QUANTIZER_TYPE::ADAPTIVE, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points);

    printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

    auto decompressor = TonSZ::BlockDecompressor<float>(TonSZ::QUANTIZER_TYPE::ADAPTIVE, std::stof(argv[3]));
    recovered_points = decompressor.decompress(compressed_points);

  }

  TonSZ::write_file(argv[4], recovered_points);
  double acc_mse = 0;
  float max_l2_error = 0;
  int overflow_count = 0;
  for (int i = 0; i < points.size(); i++) {
    // printf("points[%d]: %f, %f, %f\n", i, points[i].x(), points[i].y(), points[i].z());
    // printf("recovered_points[%d]: %f, %f, %f\n", i, recovered_points[i].x(), recovered_points[i].y(), recovered_points[i].z());
    acc_mse += (points[i].x() - recovered_points[i].x()) * (points[i].x() - recovered_points[i].x());
    acc_mse += (points[i].y() - recovered_points[i].y()) * (points[i].y() - recovered_points[i].y());
    acc_mse += (points[i].z() - recovered_points[i].z()) * (points[i].z() - recovered_points[i].z());
    // printf("distance %d: %f\n", i, (points[i] - recovered_points[i]).norm());
    if ((points[i] - recovered_points[i]).norm() > std::stof(argv[3])) {
      // printf("points[%d]: %f, %f, %f\n", i, points[i].x(), points[i].y(), points[i].z());
      // printf("recovered_points[%d]: %f, %f, %f\n", i, recovered_points[i].x(), recovered_points[i].y(), recovered_points[i].z());
      // printf("distance %d: %f\n", i, (points[i] - recovered_points[i]).norm());
      overflow_count++;
    }
    max_l2_error = std::max(max_l2_error, (points[i] - recovered_points[i]).norm());
  }
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
  return 0;
}
