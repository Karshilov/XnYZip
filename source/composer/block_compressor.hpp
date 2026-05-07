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

            BlockCompressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false, int direct_threshold = 1024) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging), direct_threshold_(direct_threshold) {}

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

                // Determine encoding mode based on quantized range
                int max_range = std::max({range_x, range_y, range_z});
                bool is_direct = (max_range < direct_threshold_);
                append_value(curve_type);
                append_value(is_direct);
                std::cout << "Encoding mode: " << (is_direct ? "direct" : "block") << std::endl;

                if (is_direct) {
                    // Direct SFC encoding: skip block division, encode global SFC deltas
                    size_t n = quantized_points.size();
                    int hilbert_bits = 0;
                    { int tmp = max_range; while (tmp > 0) { hilbert_bits++; tmp >>= 1; } }
                    if (hilbert_bits == 0) hilbert_bits = 1;

                    std::vector<std::pair<uint64_t, size_t>> sfc_entries(n);
                    for (size_t i = 0; i < n; i++) {
                        uint64_t morton = XnYZip::morton_code(quantized_points[i]);
                        uint64_t sfc = curve_type ? mve::mortonToHilbert3(morton, hilbert_bits) : morton;
                        sfc_entries[i] = {sfc, i};
                    }
                    std::sort(sfc_entries.begin(), sfc_entries.end());

                    auto copy_points = points;
                    for (size_t i = 0; i < n; i++) {
                        points[i] = copy_points[sfc_entries[i].second];
                    }

                    std::vector<uint64_t> sfc_deltas(n);
                    sfc_deltas[0] = sfc_entries[0].first;
                    for (size_t i = 1; i < n; i++) {
                        sfc_deltas[i] = sfc_entries[i].first - sfc_entries[i - 1].first;
                    }

                    auto pfd_enc = PForDeltaEncoder<uint64_t>();
                    auto pfd_result = pfd_enc.encode(sfc_deltas);

                    std::cout << "patches size of sfc_deltas: " << pfd_result.patches.size() << std::endl;
                    std::cout << "deltas size of sfc_deltas: " << pfd_result.deltas.size() << std::endl;

                    auto [meta_p, data_p] = GolombRiceCoder<uint64_t>::pack(pfd_result.patches);
                    append_value(data_p.size());
                    append_value(meta_p.size());
                    buffer.insert(buffer.end(), meta_p.begin(), meta_p.end());
                    buffer.insert(buffer.end(), data_p.begin(), data_p.end());

                    auto huff_enc = HuffmanEncoder<uint64_t>();
                    huff_enc.build(pfd_result.deltas);
                    auto huff_meta = huff_enc.get_meta();
                    auto huff_data = huff_enc.encode(pfd_result.deltas);

                    append_value(huff_data.size());
                    append_value(huff_meta.size());
                    buffer.insert(buffer.end(), huff_meta.begin(), huff_meta.end());
                    buffer.insert(buffer.end(), huff_data.begin(), huff_data.end());
                } else {
                    // Block-based encoding
                    auto [blk, cnt, quads, repos, ords, is_hilbert] = block_quantize(quantized_points, 64, 64, 64, range_x, range_y, range_z, curve_type);

                    auto copy_points = points;
                    for (size_t i = 0; i < ords.size(); i++) {
                        points[i] = copy_points[ords[i]];
                    }

                    for (int i = cnt.size() - 1; i > 0; i--) cnt[i] -= cnt[i - 1];
                    for (int i = blk.size() - 1; i > 0; i--) blk[i] -= blk[i - 1];

                    auto bitpacker = BitPacker<uint64_t>();
                    auto [meta, data] = bitpacker.pack(blk);

                    append_value(data.size());
                    append_value(meta.size());
                    buffer.insert(buffer.end(), meta.begin(), meta.end());
                    buffer.insert(buffer.end(), data.begin(), data.end());

                    auto [meta2, data2] = bitpacker.pack(cnt);

                    append_value(data2.size());
                    append_value(meta2.size());
                    buffer.insert(buffer.end(), meta2.begin(), meta2.end());
                    buffer.insert(buffer.end(), data2.begin(), data2.end());

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
                }

                append_value(offset.x());
                append_value(offset.y());
                append_value(offset.z());

                auto final_compressed = compress_u8_vector(buffer);
                std::cout << "compressed size of coords (after zstd): " << final_compressed.size() << std::endl;
                std::cout << "compression ratio of coords (after zstd): " << (double)points.size() * 3 * sizeof(float) / final_compressed.size() << std::endl;
    
                return final_compressed;
            }

        private:
            QUANTIZER_TYPE quantizer_type_;
            bool is_debugging_;
            float l2_bound_;
            int direct_threshold_;
    };

}  // namespace XnYZip

#endif // TON_SZ_BLOCK_COMPRESSOR_HPP
