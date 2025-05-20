#include <iostream>
#include <string>
#include <zstd.h>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "io/fileUtils.hpp"
#include "lib.hpp"

auto decompress_int_vector(const std::vector<uint8_t>& compressed_data, size_t original_count) -> std::vector<uint8_t> {
    size_t decompressed_size = original_count * sizeof(uint8_t);
    std::vector<uint8_t> output(original_count);

    size_t actual_size = ZSTD_decompress(
        output.data(), decompressed_size,
        compressed_data.data(), compressed_data.size()
    );

    if (ZSTD_isError(actual_size)) {
        throw std::runtime_error(ZSTD_getErrorName(actual_size));
    }

    if (actual_size != decompressed_size) {
        throw std::runtime_error("Mismatched decompressed size");
    }

    return output;
}

auto compress_int_vector(const std::vector<uint8_t>& data, int compression_level = 3) -> std::vector<uint8_t> {
    size_t input_size = data.size() * sizeof(uint8_t);
    size_t max_compressed_size = ZSTD_compressBound(input_size);

    std::vector<uint8_t> compressed(max_compressed_size);

    size_t compressed_size = ZSTD_compress(
        compressed.data(), max_compressed_size,
        data.data(), input_size,
        compression_level
    );

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(ZSTD_getErrorName(compressed_size));
    }

    compressed.resize(compressed_size);  // Trim unused space
    return compressed;
}

auto main(int argc, char* argv[]) -> int
{
  auto const file_name = argv[1];
  auto points = TonSZ::read_file<float>(file_name);
  auto const quantizer = TonSZ::TruncatedOctahedronQuantizer(std::stof(argv[2]));
  auto const quantized_points = quantizer.quantize(points);
  // auto const recovered_points = quantizer.recover(quantized_points);

  Eigen::RowVector3i offset;
  auto shifted_points = TonSZ::shift_points<int>(quantized_points, offset);
  
  std::vector<size_t> sorted_indices = TonSZ::z_order_sort_indices(shifted_points);

  TonSZ::apply_permutation_inplace<Eigen::RowVector<float, 3> >(points, sorted_indices);
  TonSZ::apply_permutation_inplace<Eigen::RowVector3i>(shifted_points, sorted_indices);

  auto delta_encoder = TonSZ::DeltaEncoder<int>();
  auto delta_encoded_points = delta_encoder.encode(shifted_points);

  std::vector<int> coordwise_values = std::vector<int>(points.size() * 3);
  for (size_t i = 0; i < points.size(); ++i) {
    coordwise_values[i * 3] = delta_encoded_points[i].x();
    coordwise_values[i * 3 + 1] = delta_encoded_points[i].y();
    coordwise_values[i * 3 + 2] = delta_encoded_points[i].z();
  }

  auto encoder = TonSZ::HuffmanEncoder<int>();
  encoder.build(coordwise_values);
  auto meta = encoder.get_meta();
  auto compressed = encoder.encode(coordwise_values);

  // for (size_t i = 0; i < point_x.size(); ++i) {
  //   if (point_x[i] != delta_encoded_points[i].x()) {
  //     std::cout << "error in x" << std::endl;
  //   }
  // }

  std::cout << "original size of coords: " << coordwise_values.size() * sizeof(int) << std::endl;
  std::cout << "compressed size of coords: " << compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t) << std::endl;
  std::cout << "compression ratio of coords: " << (double)coordwise_values.size() * sizeof(int) / (compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;

  auto compressed_int_vector = compress_int_vector(compressed);

  std::cout << "compressed size of coords (after zstd): " << compressed_int_vector.size() << std::endl;
  std::cout << "compression ratio of coords (after zstd): " << (double)coordwise_values.size() * sizeof(int) / (compressed_int_vector.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;

  TonSZ::write_file_bin<uint8_t>("compressed_coords.bin", compressed_int_vector);

  // begin recovery
  auto saved_data = TonSZ::read_file_bin<uint8_t>("compressed_coords.bin");
  auto decompressed_int_vector = decompress_int_vector(saved_data, compressed.size());

  std::cout << "Zstd decompression finished" << std::endl;
  auto decoder = TonSZ::HuffmanEncoder<int>();
  decoder.load_meta(meta);
  auto decoded = decoder.decode(decompressed_int_vector, coordwise_values.size());

  std::cout << "Huffman decoding finished" << std::endl;

  auto reconstructed_points = std::vector<Eigen::RowVector3i>(coordwise_values.size() / 3);
  for (int i = 0; i < coordwise_values.size() / 3; i++) {
    reconstructed_points[i] = Eigen::RowVector3i(decoded[i * 3], decoded[i * 3 + 1], decoded[i * 3 + 2]);
  }
  auto delta_codes = delta_encoder.decode(reconstructed_points);

  std::cout << "Delta decoding finished" << std::endl;
  auto unshifted_points = TonSZ::unshift_points(delta_codes, offset);

  std::cout << "Unshifting finished" << std::endl;
  auto recovered_points = quantizer.recover(unshifted_points);

  std::cout << "Recovering finished" << std::endl;

  double psnr, nrmse, max_diff;
  std::vector<float> coordwise_original_points(points.size() * 3);
  std::vector<float> coordwise_recovered_points(recovered_points.size() * 3);

  for (size_t i = 0; i < points.size(); ++i) {
    coordwise_original_points[i * 3] = points[i].x();
    coordwise_original_points[i * 3 + 1] = points[i].y();
    coordwise_original_points[i * 3 + 2] = points[i].z();
    coordwise_recovered_points[i * 3] = recovered_points[i].x();
    coordwise_recovered_points[i * 3 + 1] = recovered_points[i].y();
    coordwise_recovered_points[i * 3 + 2] = recovered_points[i].z();
  }
  
  
  TonSZ::verify(coordwise_original_points.data(), coordwise_recovered_points.data(), points.size(), psnr, nrmse, max_diff);

  // TonSZ::HuffmanEncoder<int> encoder;
  // encoder.build(data);
  // auto meta       = encoder.get_meta();
  // auto compressed = encoder.encode(data);

  // std::cout << "原始字节数: " 
  //           << data.size() * sizeof(int) 
  //           << "，meta 大小: " 
  //           << meta.size() 
  //           << "，压缩后大小: " 
  //           << compressed.size() 
  //           << "\n";

  // TonSZ::HuffmanEncoder<int> decoder;
  // decoder.load_meta(meta);
  // auto decoded = decoder.decode(compressed, data.size());

  // std::cout << "解码结果: ";
  // for (int v : decoded) std::cout << v << ' ';
  // std::cout << (decoded == data ? "（成功）" : "（失败）") << "\n";
  TonSZ::write_file(argv[3], recovered_points);
  return 0;
}
