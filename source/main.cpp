#include <string>
#include <zstd.h>
#include <cstring>
#include <vector>
#include "io/fileUtils.hpp"
#include "lib.hpp"

auto main(int argc, char* argv[]) -> int
{
  auto const file_name = argv[1];
  auto points = XnYSZ::read_file<float>(file_name);
  std::vector<Eigen::RowVector3f> recovered_points;

  if (argc != 6) {
    printf("Usage: %s <input_file> <quantizer_type (cube/to)> <L2 bound> <curve_type (-z/-h)> <decompression_file_name>\n", argv[0]);
    return 1;
  }

  bool is_hilbert = std::string(argv[4]) == "-h";
  std::cout << "SFC type: " << (is_hilbert ? "Hilbert" : "Z-order") << std::endl;

  if (std::string(argv[2]) == "cube") {
    XnYSZ::Timer timer;
    timer.start();
    auto compressor = XnYSZ::BlockCompressor<float>(XnYSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points, is_hilbert);

    timer.stop("XnYSZ compression");

    printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

    timer.start();
    auto decompressor = XnYSZ::BlockDecompressor<float>(XnYSZ::QUANTIZER_TYPE::CUBE, std::stof(argv[3]));
    recovered_points = decompressor.decompress(compressed_points);

    timer.stop("XnYSZ decompression");

  } else if (std::string(argv[2]) == "to") {
    XnYSZ::Timer timer;
    timer.start();
    auto compressor = XnYSZ::BlockCompressor<float>(XnYSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    auto compressed_points = compressor.compress(points, is_hilbert);

    timer.stop("XnYSZ compression");

    printf("compression ratio: %f\n", (float)points.size() * 3 * sizeof(float) / compressed_points.size() * sizeof(uint8_t));

    timer.start();
    auto decompressor = XnYSZ::BlockDecompressor<float>(XnYSZ::QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON, std::stof(argv[3]));
    recovered_points = decompressor.decompress(compressed_points);

    timer.stop("XnYSZ decompression");

    printf("decompressed finished, size: %lu\n", recovered_points.size());
  }

  XnYSZ::write_file(argv[5], recovered_points);
  return 0;
}
