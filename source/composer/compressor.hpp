#ifndef TON_SZ_COMPRESSOR_HPP
#define TON_SZ_COMPRESSOR_HPP

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <zstd.h>
#include <cstring>
#include <stdexcept>
#include "../quantizer/truncated_octahedron_quantizer.hpp"
#include "../quantizer/adaptive_quantizer.hpp"
#include "../quantizer/cube_quantizer.hpp"
#include "../io/fileUtils.hpp"
#include "../encoder/huffman_encoder.hpp"
#include "../encoder/delta_encoder.hpp"
#include "../preprocessor/z_order_curve.hpp"
#include "../preprocessor/shifting.hpp"
#include "utils.hpp"
#include <set>

namespace TonSZ {

    template<typename T>
    class Compressor {

        public:

            Compressor(QUANTIZER_TYPE quantizer_type, float l2_bound, bool is_debugging = false) : quantizer_type_(quantizer_type), l2_bound_(l2_bound), is_debugging_(is_debugging) {}

            ~Compressor() = default;

            auto compress(std::vector<Eigen::RowVector<T, 3> >& points) -> std::vector<uint8_t> {

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
                auto quantized_points = quantizer->quantize(points, params);

                // Count unique quantized points
                std::vector<Eigen::RowVector3i> unique_points;
                struct RowVector3iComparator {
                    bool operator()(const Eigen::RowVector3i& a, const Eigen::RowVector3i& b) const {
                        if(a.x() != b.x()) return a.x() < b.x();
                        if(a.y() != b.y()) return a.y() < b.y();
                        return a.z() < b.z();
                    }
                };

                std::set<Eigen::RowVector3i, RowVector3iComparator> unique_set;
                for(const auto& p : quantized_points) {
                    unique_set.insert(p);
                }

                 {
                    std::cout << "Total points: " << quantized_points.size() << std::endl;
                    std::cout << "Unique points: " << unique_set.size() << std::endl;
                    std::cout << "Duplicate ratio: " << 1.0 - (double)unique_set.size() / quantized_points.size() << std::endl;
                }
                
                std::vector<size_t> sorted_indices = z_order_sort_indices(quantized_points);

                if (is_debugging_) {
                    write_file_bin<size_t>("sorted_indices.bin", sorted_indices);
                }

                apply_permutation_inplace<Eigen::RowVector<T, 3> >(points, sorted_indices);
                apply_permutation_inplace<Eigen::RowVector<int, 3> >(quantized_points, sorted_indices);

                auto delta_encoder = DeltaEncoder<int>();
                auto delta_encoded_points = delta_encoder.encode(quantized_points);

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
                append_value(params.size());
                for (size_t i = 0; i < params.size(); ++i) {
                    append_value(params[i]);
                }

                append_value(offset.x());
                append_value(offset.y());
                append_value(offset.z());

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
