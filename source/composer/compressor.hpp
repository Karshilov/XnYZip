#ifndef TON_SZ_COMPRESSOR_HPP
#define TON_SZ_COMPRESSOR_HPP

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <zstd.h>
#include <cstring>
#include <stdexcept>
#include "../quantizer/truncated_octahedron_quantizer.hpp"
#include "../quantizer/cube_quantizer.hpp"
#include "../io/fileUtils.hpp"
#include "../encoder/huffman_encoder.hpp"
#include "../encoder/delta_encoder.hpp"
#include "../preprocessor/z_order_curve.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"
namespace TonSZ {

    auto compress_u8_vector(const std::vector<uint8_t>& data, int compression_level = 3) -> std::vector<uint8_t> {
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

    template<typename T>
    class Compressor {

        public:

            Compressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

            ~Compressor() = default;

            auto compress(std::vector<Eigen::RowVector<T, 3> >& points) -> std::vector<uint8_t> {
                std::unique_ptr<BaseQuantizer<T>> quantizer;
                if (quantizer_type_ == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
                    quantizer = std::make_unique<TruncatedOctahedronQuantizer<T>>(l2_bound_);
                } else {
                    quantizer = std::make_unique<CubeQuantizer<T>>(l2_bound_);
                }
                auto const quantized_points = quantizer->quantize(points);

                Eigen::RowVector3i offset;
                auto shifted_points = shift_points<int>(quantized_points, offset);
                
                std::vector<size_t> sorted_indices = z_order_sort_indices(shifted_points);

                if (is_debugging_) {
                    write_file_bin<size_t>("sorted_indices.bin", sorted_indices);
                }

                apply_permutation_inplace<Eigen::RowVector<T, 3> >(points, sorted_indices);
                apply_permutation_inplace<Eigen::RowVector3i>(shifted_points, sorted_indices);

                auto delta_encoder = DeltaEncoder<int>();
                auto delta_encoded_points = delta_encoder.encode(shifted_points);

                std::vector<int> coordwise_values = std::vector<int>(points.size() * 3);
                for (size_t i = 0; i < points.size(); ++i) {
                    coordwise_values[i * 3] = delta_encoded_points[i].x();
                    coordwise_values[i * 3 + 1] = delta_encoded_points[i].y();
                    coordwise_values[i * 3 + 2] = delta_encoded_points[i].z();
                }

                auto encoder = HuffmanEncoder<int>();
                encoder.build(coordwise_values);
                auto meta = encoder.get_meta();
                auto compressed = encoder.encode(coordwise_values);

                std::cout << "original size of coords: " << coordwise_values.size() * sizeof(int) << std::endl;
                std::cout << "compressed size of coords: " << compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t) << std::endl;
                std::cout << "compression ratio of coords: " << (double)coordwise_values.size() * sizeof(int) / (compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;

                std::vector<uint8_t> buffer;

                auto append_value = [&](auto v) {
                    auto const* p = reinterpret_cast<uint8_t const*>(&v);
                    buffer.insert(buffer.end(), p, p + sizeof(v));
                };

                uint64_t num_points = points.size();
                append_value(num_points);

                append_value(static_cast<int32_t>(offset.x()));
                append_value(static_cast<int32_t>(offset.y()));
                append_value(static_cast<int32_t>(offset.z()));

                uint32_t meta_size = meta.size();
                append_value(meta_size);
                buffer.insert(buffer.end(), meta.begin(), meta.end());

                uint64_t raw_size = compressed.size();
                append_value(raw_size);
    
                buffer.insert(buffer.end(), compressed.begin(), compressed.end());

                auto final_compressed = compress_u8_vector(buffer);
                std::cout << "compressed size of coords (after zstd): " << final_compressed.size() << std::endl;
                std::cout << "compression ratio of coords (after zstd): " << (double)coordwise_values.size() * sizeof(int) / (final_compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;
    
                return final_compressed;
            }

        private:
            QUANTIZER_TYPE quantizer_type_;
            bool is_debugging_;
            float l2_bound_;
    };

}

#endif // TON_SZ_COMPRESSOR_HPP
