#ifndef TON_SZ_BLOCK_DECOMPRESSOR_HPP
#define TON_SZ_BLOCK_DECOMPRESSOR_HPP

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
#include "../encoder/bitpacking_encoder.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"

namespace TonSZ {

    template<typename T>
    class BlockDecompressor {
    public:
        BlockDecompressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

        ~BlockDecompressor() = default;

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

            // std::cout << "decompressing: num_points: " << num_points << ", param_size: " << param_size << ", range_x: " << range_x << ", range_y: " << range_y << ", range_z: " << range_z << std::endl;

            size_t blk_size, meta_size;
            read_value(blk_size);
            read_value(meta_size);
            std::vector<uint8_t> meta(p, p + meta_size); p += meta_size;
            std::vector<uint8_t> data(p, p + blk_size); p += blk_size;
            auto blk = BitPacker<uint64_t>::unpack({meta, data});

            // std::cout << "decompressing: blk_size: " << blk_size << ", meta_size: " << meta_size << std::endl;

            read_value(blk_size);
            read_value(meta_size);
            meta.assign(p, p + meta_size); p += meta_size;
            data.assign(p, p + blk_size); p += blk_size;
            auto blkcnt = BitPacker<uint64_t>::unpack({meta, data});

            // std::cout << "decompressing: cnt_size: " << blk_size << ", meta_size: " << meta_size << std::endl;

            read_value(blk_size);
            read_value(meta_size);
            meta.assign(p, p + meta_size); p += meta_size;
            std::vector<uint8_t> encoded(p, p + blk_size); p += blk_size;
            auto encoder = HuffmanEncoder<uint64_t>();
            encoder.load_meta(meta);
            auto repos = encoder.decode(encoded, num_points);

            // std::cout << "decompressing: repos_size: " << blk_size << ", meta_size: " << meta_size << std::endl;

            read_value(blk_size);
            read_value(meta_size);
            meta.assign(p, p + meta_size); p += meta_size;
            encoded.assign(p, p + blk_size); p += blk_size;
            auto encoder2 = HuffmanEncoder<uint8_t>();
            encoder2.load_meta(meta);
            auto quads = encoder2.decode(encoded, num_points);

            // std::cout << "decompressing: quads_size: " << blk_size << ", meta_size: " << meta_size << std::endl;


            // recover delta
            for (int i = 1; i < blkcnt.size(); i++) blkcnt[i] += blkcnt[i - 1];
            for (int i = 1; i < blk.size(); i++) blk[i] += blk[i - 1];
            
            write_file_bin("quads-decompressed", quads);
            write_file_bin("repos-decompressed", repos);
            write_file_bin("cnt-decompressed", blkcnt);
            write_file_bin("blk-decompressed", blk);

            T ox, oy, oz;
            read_value(ox);
            read_value(oy);
            read_value(oz);
            Eigen::RowVector3<T> offset(ox, oy, oz);

            // std::cout << "decompressing: offset: " << ox << ", " << oy << ", " << oz << std::endl;

            LCPMeta lcp_meta;
            lcp_meta.blkst = blk;
            lcp_meta.blkcnt = blkcnt;
            lcp_meta.quads = quads;
            lcp_meta.repos = repos;

            auto coords = recover_from_lcp_meta<int>(lcp_meta, 128, 128, 128, range_x, range_y, range_z);

            // std::cout << "decompressing: coords_size: " << coords.size() << std::endl;

            std::unique_ptr<BaseQuantizer<T>> quantizer;
            if (quantizer_type_ == QUANTIZER_TYPE::TRUNCATED_OCTAHEDRON) {
                quantizer = std::make_unique<TruncatedOctahedronQuantizer<T>>(l2_bound_);
            } else if (quantizer_type_ == QUANTIZER_TYPE::ADAPTIVE) {
                quantizer = std::make_unique<AdaptiveQuantizer<T>>(l2_bound_);
            } else {
                quantizer = std::make_unique<CubeQuantizer<T>>(l2_bound_);
            }

            auto recovered_points = quantizer->recover(coords, params);

            // std::cout << "decompressing: recovered_points_size: " << recovered_points.size() << std::endl;

            auto unshifted = TonSZ::unshift_points(recovered_points, offset);

            // std::cout << "decompressing: unshifted_points_size: " << unshifted.size() << std::endl;

            return unshifted;
        }

    private:
        QUANTIZER_TYPE quantizer_type_;
        float l2_bound_;
        bool is_debugging_;
    };

} // namespace TonSZ

#endif // TON_SZ_BLOCK_DECOMPRESSOR_HPP
