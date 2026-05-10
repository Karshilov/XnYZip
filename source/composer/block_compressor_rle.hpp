#ifndef TON_SZ_BLOCK_COMPRESSOR_RLE_HPP
#define TON_SZ_BLOCK_COMPRESSOR_RLE_HPP

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
    class BlockCompressorRLE {

        public:

            BlockCompressorRLE(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false, int direct_threshold = 1024) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging), direct_threshold_(direct_threshold) {}

            ~BlockCompressorRLE() = default;

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

                double acc_mse = 0;
                double max_l2_error = 0;
                int overflow_count = 0;
                auto recovered_points = quantizer->recover(quantized_points, params);
                for (int i = 0; i < shifted_points.size(); i++) {
                  // printf("points[%d]: %f, %f, %f\n", i, points[i].x(), points[i].y(), points[i].z());
                  // printf("recovered_points[%d]: %f, %f, %f\n", i, recovered_points[i].x(), recovered_points[i].y(), recovered_points[i].z());
                  double mse = ((shifted_points[i].x() - recovered_points[i].x()) * (shifted_points[i].x() - recovered_points[i].x()))+
                  ((shifted_points[i].y() - recovered_points[i].y()) * (shifted_points[i].y() - recovered_points[i].y()))+
                  ((shifted_points[i].z() - recovered_points[i].z()) * (shifted_points[i].z() - recovered_points[i].z()));
                  acc_mse += mse;
                  // printf("distance %d: %f\n", i, (points[i] - recovered_points[i]).norm());
                    max_l2_error = std::max(max_l2_error, sqrt(mse));
                }
                printf("Max L2 error: %lf\n", max_l2_error);
                acc_mse /= shifted_points.size();
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

                std::vector<uint8_t> buffer;

                auto append_value = [&](auto v) {
                    auto const* p = reinterpret_cast<uint8_t const*>(&v);
                    buffer.insert(buffer.end(), p, p + sizeof(v));
                };

                // Count unique quantized points
                std::vector<Eigen::RowVector3i> unique_points;
                struct RowVector3iHasher {
                    std::size_t operator()(const Eigen::RowVector3i& v) const {
                        uint64_t h = (uint64_t)v[0] * 73856093u ^
                        (uint64_t)v[1] * 19349663u ^
                        (uint64_t)v[2] * 83492791u;
                        return h;
                    }
                };

                std::unordered_map<Eigen::RowVector3i, int, RowVector3iHasher> unique_map;
                size_t original_size = quantized_points.size();
                for(const auto& p : quantized_points) {
                    unique_map[p]++;
                }

                quantized_points.clear();
                int dup_cnt = 0;
                std::vector<int> cnts;
                for(const auto& [p, cnt] : unique_map) {
                    dup_cnt += cnt;
                    cnts.push_back(cnt);
                    quantized_points.push_back(p);
                }

                printf("Unique points: %lu\n", unique_map.size());
                printf("Duplicate ratio: %.3f\n", 1.0 - (double)unique_map.size() / original_size);

                uint64_t num_points = unique_map.size();
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

                    auto copy_cnts = cnts;
                    for (size_t i = 0; i < n; i++) {
                        cnts[i] = copy_cnts[sfc_entries[i].second];
                    }

                    quantized_points.clear();

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

                    auto copy_cnts = cnts;
                    for (int i = 0; i < cnts.size(); i++) {
                        cnts[i] = copy_cnts[ords[i]];
                    }

                    quantized_points.clear();

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

                    auto [meta_p, data_p] = GolombRiceCoder<uint64_t>::pack(p_for_delta_data.patches);

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

                auto encoder3 = HuffmanEncoder<int>();

                // for (int i = cnts.size() - 1; i > 0; i--) cnts[i] -= cnts[i - 1];

                encoder3.build(cnts);
                auto meta5 = encoder3.get_meta();
                auto compressed5 = encoder3.encode(cnts);

                if (static_cast<double>(meta5.size()) > static_cast<double>(compressed5.size()) * 0.25) {
                    encoder3.set_use_exp_golomb(true);
                }
                encoder3.build(cnts);
                meta5 = encoder3.get_meta();
                compressed5 = encoder3.encode(cnts);

                append_value(compressed5.size());
                append_value(meta5.size());
                buffer.insert(buffer.end(), meta5.begin(), meta5.end());
                buffer.insert(buffer.end(), compressed5.begin(), compressed5.end());

                std::cout << "cnts size: " << compressed5.size() << ", meta size: " << meta5.size() << std::endl;

                append_value(offset.x());
                append_value(offset.y());
                append_value(offset.z());

                auto final_compressed = compress_u8_vector(buffer);
                std::cout << "compressed size of coords (after zstd): " << final_compressed.size() << std::endl;
                std::cout << "compression ratio of coords (after zstd): " << (double)points.size() * 3 * sizeof(float) / final_compressed.size() << std::endl;

                return final_compressed;
            }

            // === Distributed-RLE entry point ===
            //
            // Used by the MPI driver's `--mode mapreduce` flow. Takes
            // *already-deduplicated* (cells, counts) — typically produced by a
            // distributed Map+Shuffle+Reduce phase that aggregates each global
            // cell's total count across ranks — and runs the rest of the RLE
            // encoding pipeline (SFC sort + delta + bitpack + Huffman + zstd
            // for the cell stream and the count stream).
            //
            // `range_x/y/z` and `offset` should be the GLOBAL bbox of the
            // dataset (not chunk-local), broadcast from rank 0. Using global
            // bbox guarantees every rank emits a consistent sub-block grid.
            //
            // Forces block mode (always); skips the direct-mode path that has
            // the pre-existing 1-point overflow bug and would also produce
            // inconsistent encodings across ranks.
            //
            // Existing compress() above is unchanged.
            auto encode_cells_with_counts(
                std::vector<Eigen::RowVector3i>& cells,
                std::vector<int>& counts,
                int range_x, int range_y, int range_z,
                Eigen::RowVector3<T> offset,
                bool curve_type)
                -> std::vector<uint8_t>
            {
                if (cells.size() != counts.size()) {
                    throw std::runtime_error("encode_cells_with_counts: cells and counts size mismatch");
                }

                std::vector<uint8_t> buffer;
                auto append_value = [&](auto v) {
                    auto const* p = reinterpret_cast<uint8_t const*>(&v);
                    buffer.insert(buffer.end(), p, p + sizeof(v));
                };

                uint64_t num_points = cells.size();
                append_value(num_points);
                append_value(static_cast<uint64_t>(0)); // params.size() == 0 for TO/cube quantizer in distributed mode

                append_value(range_x);
                append_value(range_y);
                append_value(range_z);

                bool is_direct = false;  // always block mode in distributed
                append_value(curve_type);
                append_value(is_direct);

                // ---- Block-mode encoding path (mirrors compress() else branch) ----
                auto [blk, cnt_blk, quads, repos, ords, is_hilbert] =
                    block_quantize(cells, 64, 64, 64, range_x, range_y, range_z, curve_type);

                // Reorder counts via ords (matches what compress() does)
                {
                    auto copy_cnts = counts;
                    for (size_t i = 0; i < ords.size(); i++) {
                        counts[i] = copy_cnts[ords[i]];
                    }
                }

                cells.clear();

                for (int i = cnt_blk.size() - 1; i > 0; i--) cnt_blk[i] -= cnt_blk[i - 1];
                for (int i = blk.size() - 1; i > 0; i--) blk[i] -= blk[i - 1];

                auto bitpacker = BitPacker<uint64_t>();
                {
                    auto [meta, data] = bitpacker.pack(blk);
                    append_value(data.size());
                    append_value(meta.size());
                    buffer.insert(buffer.end(), meta.begin(), meta.end());
                    buffer.insert(buffer.end(), data.begin(), data.end());
                }
                {
                    auto [meta2, data2] = bitpacker.pack(cnt_blk);
                    append_value(data2.size());
                    append_value(meta2.size());
                    buffer.insert(buffer.end(), meta2.begin(), meta2.end());
                    buffer.insert(buffer.end(), data2.begin(), data2.end());
                }

                {
                    auto p_for_delta_encoder = PForDeltaEncoder<uint64_t>();
                    auto p_for_delta_data = p_for_delta_encoder.encode(repos);
                    auto [meta_p, data_p] = GolombRiceCoder<uint64_t>::pack(p_for_delta_data.patches);
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
                }

                {
                    auto p_for_delta_encoder2 = PForDeltaEncoder<uint8_t>();
                    auto p_for_delta_data2 = p_for_delta_encoder2.encode(quads);
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

                // Encode counts (mirrors compress() lines 290-310)
                {
                    auto encoder3 = HuffmanEncoder<int>();
                    encoder3.build(counts);
                    auto meta5 = encoder3.get_meta();
                    auto compressed5 = encoder3.encode(counts);
                    if (static_cast<double>(meta5.size()) > static_cast<double>(compressed5.size()) * 0.25) {
                        encoder3.set_use_exp_golomb(true);
                    }
                    encoder3.build(counts);
                    meta5 = encoder3.get_meta();
                    compressed5 = encoder3.encode(counts);
                    append_value(compressed5.size());
                    append_value(meta5.size());
                    buffer.insert(buffer.end(), meta5.begin(), meta5.end());
                    buffer.insert(buffer.end(), compressed5.begin(), compressed5.end());
                }

                append_value(offset.x());
                append_value(offset.y());
                append_value(offset.z());

                auto final_compressed = compress_u8_vector(buffer);
                std::cout << "[mapreduce] encode_cells_with_counts: cells=" << num_points
                          << " pre_zstd=" << buffer.size()
                          << " post_zstd=" << final_compressed.size() << std::endl;
                return final_compressed;
            }

        private:
            QUANTIZER_TYPE quantizer_type_;
            bool is_debugging_;
            float l2_bound_;
            int direct_threshold_;

    };
}

#endif // TON_SZ_BLOCK_COMPRESSOR_RLE_HPP
