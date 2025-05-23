#ifndef TON_SZ_DECOMPRESSOR_HPP
#define TON_SZ_DECOMPRESSOR_HPP

#include <vector>
#include <Eigen/Dense>
#include <zstd.h>
#include <cstring>
#include <stdexcept>
#include "../quantizer/truncated_octahedron_quantizer.hpp"
#include "../quantizer/cube_quantizer.hpp"
#include "../quantizer/adaptive_quantizer.hpp"
#include "../io/fileUtils.hpp"
#include "../encoder/huffman_encoder.hpp"
#include "../encoder/delta_encoder.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"
namespace TonSZ {
    
    template<typename T>
    class Decompressor {
    public:
        Decompressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

        ~Decompressor() = default;

        auto decompress(const std::vector<uint8_t>& compressed_data) -> std::vector<Eigen::RowVector<T, 3>> {
            
            auto buffer = decompress_buffer(compressed_data);
            size_t cursor = 0;

            auto read = [&](auto& v) {
                std::memcpy(&v, buffer.data() + cursor, sizeof(v));
                cursor += sizeof(v);
            };

            uint64_t num_points = 0;
            read(num_points);
            uint64_t num_params = 0;
            read(num_params);
            std::vector<T> params(num_params);
            for (size_t i = 0; i < num_params; ++i) {
                read(params[i]);
            }

            T offx = 0;
            T offy = 0;
            T offz = 0;
            read(offx); read(offy); read(offz);
            Eigen::RowVector3<T> offset(offx, offy, offz);

            uint32_t meta_size = 0;
            read(meta_size);
            std::vector<uint8_t> meta(buffer.begin() + cursor,
                                      buffer.begin() + cursor + meta_size);
            cursor += meta_size;

            uint64_t code_size = 0;
            read(code_size);

            std::vector<uint8_t> code_stream(
                buffer.begin() + cursor,
                buffer.begin() + cursor + code_size
            );
            cursor += code_size;

            if (is_debugging_) {
                write_file_bin<uint8_t>("decompressed.raw", buffer);
            }

            TonSZ::HuffmanEncoder<int> huff;
            huff.load_meta(meta);
            size_t total_ints = num_points * 3;
            auto decoded_ints = huff.decode(code_stream, total_ints);

            std::vector<Eigen::RowVector3i> coords_i;
            coords_i.reserve(num_points);
            for (size_t i = 0; i < num_points; ++i) {
                coords_i.emplace_back(
                    decoded_ints[i*3 + 0],
                    decoded_ints[i*3 + 1],
                    decoded_ints[i*3 + 2]
                );
            }

            TonSZ::DeltaEncoder<int> delta_decoder;
            auto undelta = delta_decoder.decode(coords_i);

            std::unique_ptr<BaseQuantizer<T>> quantizer;
            if (quantizer_type_ == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
                quantizer = std::make_unique<TruncatedOctahedronQuantizer<T>>(l2_bound_);
            } else if (quantizer_type_ == QUANTIZER_TYPE::ADAPTIVE) {
                quantizer = std::make_unique<AdaptiveQuantizer<T>>(l2_bound_);
            } else {
                quantizer = std::make_unique<CubeQuantizer<T>>(l2_bound_);
            }
            auto recovered = quantizer->recover(undelta, params);

            auto unshifted = TonSZ::unshift_points(recovered, offset);

            return unshifted;
        }

    private:
        QUANTIZER_TYPE quantizer_type_;
        float l2_bound_;
        bool is_debugging_;
    };

}

#endif // TON_SZ_DECOMPRESSOR_HPP
