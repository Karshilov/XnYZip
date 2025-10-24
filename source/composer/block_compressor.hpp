#ifndef TON_SZ_BLOCK_COMPRESSOR_HPP
#define TON_SZ_BLOCK_COMPRESSOR_HPP

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <zstd.h>
#include <cstring>
#include <stdexcept>
#include "../quantizer/truncated_octahedron_quantizer.hpp"
#include "../quantizer/adaptive_quantizer.hpp"
#include "../quantizer/cube_quantizer.hpp"
#include "../quantizer/block_quantizer.hpp"
#include "../io/fileUtils.hpp"
#include "../encoder/huffman_encoder.hpp"
#include "../encoder/delta_encoder.hpp"
#include "../encoder/p_for_delta_encoder.hpp"
#include "../encoder/bitpacking_encoder.hpp"
#include "../encoder/golomb_rice_encoder.hpp"
#include "../preprocessor/z_order_curve.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"

namespace XnYZip {

    template<typename T>
    class BlockCompressor {

        public:

            BlockCompressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

            ~BlockCompressor() = default;

            auto compress(std::vector<Eigen::RowVector<T, 3> >& points, bool curve_type) -> std::vector<uint8_t> {
                Eigen::RowVector3<T> offset;
                auto const shifted_points = shift_points<T>(points, offset);
                
                std::unique_ptr<BaseQuantizer<T>> quantizer;
                if (quantizer_type_ == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
                    quantizer = std::make_unique<TruncatedOctahedronQuantizer<T>>(l2_bound_);
                } else if (quantizer_type_ == QUANTIZER_TYPE::ADAPTIVE) {
                    quantizer = std::make_unique<AdaptiveQuantizer<T>>(l2_bound_);
                } else {
                    quantizer = std::make_unique<CubeQuantizer<T>>(l2_bound_);
                }
                std::vector<T> params;
                auto quantized_points = quantizer->quantize(shifted_points, params);

                std::vector<uint8_t> buffer;

                auto append_value = [&](auto v) {
                    auto const* p = reinterpret_cast<uint8_t const*>(&v);
                    buffer.insert(buffer.end(), p, p + sizeof(v));
                };

                uint64_t num_points = points.size();
                append_value(num_points);
                append_value(params.size());
                for (size_t i = 0; i < params.size(); ++i) {
                    append_value(params[i]);
                }


                int range_x = 0;
                int range_y = 0;
                int range_z = 0;

                for (const auto& p : quantized_points) {
                    range_x = std::max(range_x, p.x());
                    range_y = std::max(range_y, p.y());
                    range_z = std::max(range_z, p.z());
                }

                append_value(range_x);
                append_value(range_y);
                append_value(range_z);

                auto [blk, cnt, quads, repos, ords, is_hilbert] = block_quantize(quantized_points, 64, 64, 64, range_x, range_y, range_z, curve_type);

                auto copy_points = points;
                for (size_t i = 0; i < ords.size(); i++) {
                    points[i] = copy_points[ords[i]];
                }

                for (int i = cnt.size() - 1; i > 0; i--) cnt[i] -= cnt[i - 1];

                for (int i = blk.size() - 1; i > 0; i--) {
                    blk[i] -= blk[i - 1];
                }
                append_value(is_hilbert);

                auto bitpacker = BitPacker<uint64_t>();
                auto [meta, data] = bitpacker.pack(blk);

                append_value(data.size());
                append_value(meta.size());
                buffer.insert(buffer.end(), meta.begin(), meta.end());
                buffer.insert(buffer.end(), data.begin(), data.end());

                // std::cout << "blk size: " << data.size() << ", meta size: " << meta.size() << std::endl;

                auto [meta2, data2] = bitpacker.pack(cnt);

                append_value(data2.size());
                append_value(meta2.size());
                buffer.insert(buffer.end(), meta2.begin(), meta2.end());
                buffer.insert(buffer.end(), data2.begin(), data2.end());

                // std::cout << "cnt size: " << data2.size() << ", meta size: " << meta2.size() << std::endl;


                auto p_for_delta_encoder = PForDeltaEncoder<uint64_t>();
                auto p_for_delta_data = p_for_delta_encoder.encode(repos);

                std::cout << "patches size of repos: " << p_for_delta_data.patches.size() << std::endl;
                std::cout << "deltas size of repos: " << p_for_delta_data.deltas.size() << std::endl;

                auto golomb_rice_encoder = GolombRiceCoder<uint64_t>();
                auto [meta_p, data_p] = golomb_rice_encoder.pack(p_for_delta_data.patches);

                append_value(data_p.size());
                append_value(meta_p.size());
                buffer.insert(buffer.end(), meta_p.begin(), meta_p.end());
                buffer.insert(buffer.end(), data_p.begin(), data_p.end());

                auto encoder = HuffmanEncoder<uint64_t>();
                encoder.build(p_for_delta_data.deltas);
                auto meta3 = encoder.get_meta();
                auto compressed = encoder.encode(p_for_delta_data.deltas);

                append_value(compressed.size());
                append_value(meta3.size());
                buffer.insert(buffer.end(), meta3.begin(), meta3.end());
                buffer.insert(buffer.end(), compressed.begin(), compressed.end());

                // std::cout << "repos size: " << compressed.size() << ", meta size: " << meta3.size() << std::endl;

                auto p_for_delta_encoder2 = PForDeltaEncoder<uint8_t>();
                auto p_for_delta_data2 = p_for_delta_encoder2.encode(quads);

                std::cout << "patches size of quads: " << p_for_delta_data2.patches.size() << std::endl;
                std::cout << "deltas size of quads: " << p_for_delta_data2.deltas.size() << std::endl;

                auto golomb_rice_encoder2 = GolombRiceCoder<uint8_t>();
                auto [meta_p2, data_p2] = golomb_rice_encoder2.pack(p_for_delta_data2.patches);

                append_value(data_p2.size());
                append_value(meta_p2.size());
                buffer.insert(buffer.end(), meta_p2.begin(), meta_p2.end());
                buffer.insert(buffer.end(), data_p2.begin(), data_p2.end());

                auto encoder2 = HuffmanEncoder<uint8_t>();
                encoder2.build(p_for_delta_data2.deltas);
                auto meta4 = encoder2.get_meta();
                auto compressed4 = encoder2.encode(p_for_delta_data2.deltas);

                append_value(compressed4.size());
                append_value(meta4.size());
                buffer.insert(buffer.end(), meta4.begin(), meta4.end());
                buffer.insert(buffer.end(), compressed4.begin(), compressed4.end());

                // std::cout << "quads size: " << compressed4.size() << ", meta size: " << meta4.size() << std::endl;

                // std::cout << "quads length: " << quads.size() << std::endl;
                // std::cout << "repos length: " << repos.size() << std::endl;
                // std::cout << "cnt length: " << cnt.size() << std::endl;
                // std::cout << "blk length: " << blk.size() << std::endl;

                // std::cout << "original size of coords: " << points.size() * 3 * sizeof(int) << std::endl;
                // std::cout << "compressed size of coords: " << compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t) << std::endl;
                // std::cout << "compression ratio of coords: " << (double)points.size() * 3 * sizeof(int) / (compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;

                append_value(offset.x());
                append_value(offset.y());
                append_value(offset.z());

                auto final_compressed = compress_u8_vector(buffer);
                std::cout << "compressed size of coords (after zstd): " << final_compressed.size() << std::endl;
                std::cout << "compression ratio of coords (after zstd): " << (double)points.size() * 3 * sizeof(int) / (final_compressed.size() * sizeof(uint8_t) + meta.size() * sizeof(uint8_t)) << std::endl;
    
                return final_compressed;
            }

        private:
            QUANTIZER_TYPE quantizer_type_;
            bool is_debugging_;
            float l2_bound_;
    };

}  // namespace XnYZip

#endif // TON_SZ_BLOCK_COMPRESSOR_HPP
