#ifndef TRUNCATED_OCTAHEDRON_QUANTIZER_HPP
#define TRUNCATED_OCTAHEDRON_QUANTIZER_HPP

#include "base_quantizer.hpp"
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>

namespace XnYSZ {

    template<typename T>
    const auto is_integer = [](T const& x) -> bool {
        return std::fabs(x - std::round(x)) < 1e-6;
    };

    template<typename T>
    class TruncatedOctahedronQuantizer : public BaseQuantizer<T> {
        public:
            explicit TruncatedOctahedronQuantizer(T scale): BaseQuantizer<T>(scale) {}

        auto quantize(std::vector<Eigen::RowVector<T, 3>> const& points, std::vector<T>& params) const -> std::vector<Eigen::RowVector3i> {
            std::vector<Eigen::RowVector3i> quantized_points {};
            quantized_points.reserve(points.size());

            for (const auto& pt : points) {

                // Eigen::RowVector<T, 3> uvw = transform_point<T>(pt, get_inverse_transform_matrix<T>(TRUNC_OCT_SCALE_)) + Eigen::RowVector<T, 3>::Constant(0.5);
                Eigen::RowVector<T, 3> quantized_xyz = pt / (2 * this->TRUNC_OCT_SCALE_ / sqrt(5));
                
                auto nearest_lattice = [&](const Eigen::RowVector3f &p) -> Eigen::RowVector3i {
                    auto nearest_even = [](float v) -> int {
                        int lo = int(std::floor(v));
                        if (lo & 1) {
                            lo -= 1;
                        }
                        int hi = lo + 2;
                        return (std::fabs(v - lo) <= std::fabs(hi - v)) ? lo : hi;
                    };
                    auto nearest_odd = [](float v) -> int {
                        int lo = int(std::floor(v));
                        if ((lo & 1) == 0) {
                            lo -= 1;
                        }
                        int hi = lo + 2;
                        return (std::fabs(v - lo) <= std::fabs(hi - v)) ? lo : hi;
                    };

                    Eigen::RowVector3i E, O;
                    for (int i = 0; i < 3; ++i) {
                        E[i] = nearest_even(p[i]);
                        O[i] = nearest_odd(p[i]);
                    }

                    float dE = (p - E.cast<float>()).squaredNorm();
                    float dO = (p - O.cast<float>()).squaredNorm();


                    if (std::min(dE, dO) > 1.25) {
                        throw std::runtime_error("Quantized point is too far from the original point");
                    }

                    return (dE <= dO) ? E : O;
                };

                auto node = nearest_lattice(quantized_xyz);
                if (node[0] & 1) {
                    node[0] += 1;
                    node[0] >>= 1;
                    node[1] += 1;
                    node[1] >>= 1;
                } else {
                    node[0] >>= 1;
                    node[1] >>= 1;
                }
                quantized_points.push_back(node);
            }

            return quantized_points;
        }

        auto recover(std::vector<Eigen::RowVector3i> const& points, std::vector<T> const& params) const -> std::vector<Eigen::RowVector<T, 3>> {
            std::vector<Eigen::RowVector<T, 3>> recovered_points {};
            recovered_points.reserve(points.size());

            for (const auto& voxel : points) {
                auto node = voxel;
                if (node[2] & 1) {
                    node[0] <<= 1;
                    node[0] -= 1;
                    node[1] <<= 1;
                    node[1] -= 1;
                } else {
                    node[0] <<= 1;
                    node[1] <<= 1;
                }
                recovered_points.push_back(node.array().template cast<T>() * 2 * this->TRUNC_OCT_SCALE_ / sqrt(5));
            }

            return recovered_points;
        }

    private:
    };

} // namespace XnYSZ

#endif // TRUNCATED_OCTAHEDRON_QUANTIZER_HPP