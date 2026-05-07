#ifndef TON_SZ_BLOCK_DECOMPRESSOR_RLE_HPP
#define TON_SZ_BLOCK_DECOMPRESSOR_RLE_HPP

#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include <zstd.h>
#include <cstring>
#include <stdexcept>
#include "../quantizer/truncated_octahedron_quantizer.hpp"
#include "../quantizer/cube_quantizer.hpp"
#include "../quantizer/block_quantizer.hpp"
#include "../quantizer/adaptive_quantizer.hpp"
#include "../io/fileUtils.hpp"
#include "../encoder/huffman_encoder.hpp"
#include "../encoder/golomb_rice_encoder.hpp"
#include "../encoder/bitpacking_encoder.hpp"
#include "../encoder/p_for_delta_encoder.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"

namespace XnYZip {

    template<typename T>
    class BlockDecompressorRLE {
    public:
        BlockDecompressorRLE(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

        ~BlockDecompressorRLE() = default;

        auto decompress(const std::vector<uint8_t>& compressed_data) -> std::vector<Eigen::RowVector<T, 3>> {
            std::vector<uint8_t> buffer = decompress_buffer(compressed_data);
            const uint8_t* p = buffer.data();

            auto read_value = [&](auto& v) {
                std::memcpy(&v, p, sizeof(v));
                p += sizeof(v);
            };

            uint64_t num_points;
            read_value(num_points);
            uint64_t param_size;
            read_value(param_size);
            std::vector<T> params(param_size);
            for (uint64_t i = 0; i < param_size; ++i) read_value(params[i]);

            int32_t range_x, range_y, range_z;
            read_value(range_x);
            read_value(range_y);
            read_value(range_z);

            bool is_hilbert;
            read_value(is_hilbert);

            bool is_direct;
            read_value(is_direct);

            // std::cout << "decompressing: num_points: " << num_points << ", param_size: " << param_size << ", range_x: " << range_x << ", range_y: " << range_y << ", range_z: " << range_z << std::endl;

            std::vector<Eigen::RowVector3i> coords;

            if (is_direct) {
                // Direct SFC decoding
                size_t blk_size, meta_size;
                read_value(blk_size);
                read_value(meta_size);
                std::vector<uint8_t> meta(p, p + meta_size); p += meta_size;
                std::vector<uint8_t> data(p, p + blk_size); p += blk_size;
                auto patches = GolombRiceCoder<uint64_t>::unpack({.meta=meta, .data=data});

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                std::vector<uint8_t> encoded(p, p + blk_size); p += blk_size;
                auto encoder = HuffmanEncoder<uint64_t>();
                encoder.load_meta(meta);
                auto deltas = encoder.decode(encoded, num_points);

                auto p_for_delta_encoder = PForDeltaEncoder<uint64_t>();
                PForDeltaData<uint64_t> p_for_delta_data;
                p_for_delta_data.patches = patches;
                p_for_delta_data.deltas = deltas;
                auto sfc_deltas = p_for_delta_encoder.decode(p_for_delta_data);

                // Prefix sum to recover SFC codes
                std::vector<uint64_t> sfc_codes(num_points);
                sfc_codes[0] = sfc_deltas[0];
                for (size_t i = 1; i < num_points; i++) {
                    sfc_codes[i] = sfc_codes[i - 1] + sfc_deltas[i];
                }

                // Decode SFC codes to coordinates
                int max_range = std::max({range_x, range_y, range_z});
                int hilbert_bits = 0;
                { int tmp = max_range; while (tmp > 0) { hilbert_bits++; tmp >>= 1; } }
                if (hilbert_bits == 0) hilbert_bits = 1;

                coords.resize(num_points);
                for (size_t i = 0; i < num_points; i++) {
                    uint64_t morton = is_hilbert ? mve::hilbertToMorton3(sfc_codes[i], hilbert_bits) : sfc_codes[i];
                    coords[i] = XnYZip::decode_morton_code(morton);
                }
            } else {
                // Block-based decoding
                size_t blk_size, meta_size;
                read_value(blk_size);
                read_value(meta_size);
                std::vector<uint8_t> meta(p, p + meta_size); p += meta_size;
                std::vector<uint8_t> data(p, p + blk_size); p += blk_size;
                auto blk = BitPacker<uint64_t>::unpack({.meta=meta, .data=data});

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                data.assign(p, p + blk_size); p += blk_size;
                auto blkcnt = BitPacker<uint64_t>::unpack({.meta=meta, .data=data});

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                data.assign(p, p + blk_size); p += blk_size;
                auto patches = GolombRiceCoder<uint64_t>::unpack({.meta=meta, .data=data});

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                std::vector<uint8_t> encoded(p, p + blk_size); p += blk_size;
                auto encoder = HuffmanEncoder<uint64_t>();
                encoder.load_meta(meta);
                auto deltas = encoder.decode(encoded, num_points);

                auto p_for_delta_encoder = PForDeltaEncoder<uint64_t>();
                PForDeltaData<uint64_t> p_for_delta_data;
                p_for_delta_data.patches = patches;
                p_for_delta_data.deltas = deltas;
                auto repos = p_for_delta_encoder.decode(p_for_delta_data);

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                data.assign(p, p + blk_size); p += blk_size;
                auto patches2 = GolombRiceCoder<uint8_t>::unpack({.meta=meta, .data=data});

                read_value(blk_size);
                read_value(meta_size);
                meta.assign(p, p + meta_size); p += meta_size;
                encoded.assign(p, p + blk_size); p += blk_size;
                auto encoder2 = HuffmanEncoder<uint8_t>();
                encoder2.load_meta(meta);
                auto deltas2 = encoder2.decode(encoded, num_points);

                auto p_for_delta_encoder2 = PForDeltaEncoder<uint8_t>();
                PForDeltaData<uint8_t> p_for_delta_data2;
                p_for_delta_data2.patches = patches2;
                p_for_delta_data2.deltas = deltas2;
                auto quads = p_for_delta_encoder2.decode(p_for_delta_data2);

                // recover delta
                for (int i = 1; i < blkcnt.size(); i++) blkcnt[i] += blkcnt[i - 1];
                for (int i = 1; i < blk.size(); i++) blk[i] += blk[i - 1];

                LCPMeta lcp_meta;
                lcp_meta.blkst = blk;
                lcp_meta.blkcnt = blkcnt;
                lcp_meta.quads = quads;
                lcp_meta.repos = repos;
                lcp_meta.is_hilbert = is_hilbert;

                coords = recover_from_lcp_meta<int>(lcp_meta, 64, 64, 64, range_x, range_y, range_z);
            }

            size_t blk_size, meta_size;
            read_value(blk_size);
            read_value(meta_size);
            std::vector<uint8_t> meta(p, p + meta_size); p += meta_size;
            std::vector<uint8_t> encoded(p, p + blk_size); p += blk_size;
            auto encoder3 = HuffmanEncoder<int>();
            encoder3.load_meta(meta);
            auto cnts = encoder3.decode(encoded, num_points);

            std::vector<Eigen::RowVector3i> full_coords;
            for (int i = 0; i < coords.size(); i++) {
                for (int j = 0; j < cnts[i]; j++) {
                    full_coords.push_back(coords[i]);
                }
            }

            std::unique_ptr<BaseQuantizer<T>> quantizer;
            if (quantizer_type_ == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
                quantizer = std::make_unique<TruncatedOctahedronQuantizer<T>>(l2_bound_);
            } else {
                quantizer = std::make_unique<CubeQuantizer<T>>(l2_bound_);
            }

            auto recovered_points = quantizer->recover(full_coords, params);

            T ox, oy, oz;
            read_value(ox);
            read_value(oy);
            read_value(oz);
            Eigen::RowVector3<T> offset(ox, oy, oz);

            auto unshifted = XnYZip::unshift_points(recovered_points, offset);

            return unshifted;
        }

    private:
        QUANTIZER_TYPE quantizer_type_;
        float l2_bound_;
        bool is_debugging_;
    };

} // namespace XnYZip

#endif // TON_SZ_BLOCK_DECOMPRESSOR_RLE_HPP
