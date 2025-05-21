#include <string>
#include <zstd.h>
#include <cstring>
#include <vector>
#include "lib.hpp"

auto main(int argc, char* argv[]) -> int
{
  auto const file_name = argv[1];
  auto points = TonSZ::read_file<float>(file_name);
  std::vector<Eigen::RowVector3f> recovered_points;
  if (std::string(argv[2]) == "cube") {
    auto compressor = TonSZ::Compressor<float>(TonSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points);

    printf("Compressed points size: %zu\n", compressed_points.size() * sizeof(uint8_t));
    printf("Points size: %zu\n", points.size() * 3 * sizeof(float));

    std::string compressed_path = std::string(argv[1]) + ".ton";
    TonSZ::write_file_bin(compressed_path, compressed_points);

    auto stream = TonSZ::read_file_bin<uint8_t>(compressed_path);
    auto decompressor = TonSZ::Decompressor<float>(TonSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    recovered_points = decompressor.decompress(stream);
  } else if (std::string(argv[2]) == "octa") {
    auto compressor = TonSZ::Compressor<float>(TonSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points);

    printf("Compressed points size: %zu\n", compressed_points.size() * sizeof(uint8_t));
    printf("Points size: %zu\n", points.size() * 3 * sizeof(float));

    std::string compressed_path = std::string(argv[1]) + ".ton";
    TonSZ::write_file_bin(compressed_path, compressed_points);

    auto stream = TonSZ::read_file_bin<uint8_t>(compressed_path);

    auto decompressor = TonSZ::Decompressor<float>(TonSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    recovered_points = decompressor.decompress(stream);
  }
 
  double psnr, nrmse, max_diff;
  std::vector<float> coordwise_original_points_x(points.size());
  std::vector<float> coordwise_original_points_y(points.size());
  std::vector<float> coordwise_original_points_z(points.size());
  std::vector<float> coordwise_recovered_points_x(recovered_points.size());
  std::vector<float> coordwise_recovered_points_y(recovered_points.size());
  std::vector<float> coordwise_recovered_points_z(recovered_points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    coordwise_original_points_x[i] = points[i].x();
    coordwise_original_points_y[i] = points[i].y();
    coordwise_original_points_z[i] = points[i].z();
    coordwise_recovered_points_x[i] = recovered_points[i].x();
    coordwise_recovered_points_y[i] = recovered_points[i].y();
    coordwise_recovered_points_z[i] = recovered_points[i].z();
  }
  
  
  TonSZ::verify(coordwise_original_points_x.data(), coordwise_recovered_points_x.data(), points.size(), psnr, nrmse, max_diff);
  TonSZ::verify(coordwise_original_points_y.data(), coordwise_recovered_points_y.data(), points.size(), psnr, nrmse, max_diff);
  TonSZ::verify(coordwise_original_points_z.data(), coordwise_recovered_points_z.data(), points.size(), psnr, nrmse, max_diff);

  TonSZ::write_file(argv[4], recovered_points);
  return 0;
}
