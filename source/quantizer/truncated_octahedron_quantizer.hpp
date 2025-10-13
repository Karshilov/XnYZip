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

            Eigen::RowVector<T, 3> ones = Eigen::RowVector<T, 3>::Ones();

            for (const auto& pt : points) {

                Eigen::RowVector<T, 3> quantized_xyz = pt / (2 * this->TRUNC_OCT_SCALE_ / sqrt(5));

                auto nearest_lattice_optimized = [&](const Eigen::RowVector<T, 3> &p) -> Eigen::RowVector3i {
                    Eigen::RowVector<T, 3> E_float = (p * 0.5).array().round().matrix() * 2.0;
                    Eigen::RowVector<T, 3> O_float = (((p - ones) * 0.5).array().round().matrix() * 2.0) + ones;
                
                    T dE = (p - E_float).squaredNorm();
                    T dO = (p - O_float).squaredNorm();
                
                    return (dE <= dO) ? E_float.template cast<int>() : O_float.template cast<int>();
                };

                auto node = nearest_lattice_optimized(quantized_xyz);
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