#ifndef TRUNCATED_OCTAHEDRON_QUANTIZER_HPP
#define TRUNCATED_OCTAHEDRON_QUANTIZER_HPP

#include "base_quantizer.hpp"
#include <vector>
#include <Eigen/Dense>

namespace TonSZ {

    template<typename T>
    const auto is_integer = [](T const& x) -> bool {
        return std::fabs(x - std::round(x)) < 1e-6;
    };

    template<typename T>
    class TruncatedOctahedronQuantizer : public BaseQuantizer<T> {
        public:
            explicit TruncatedOctahedronQuantizer(T scale): BaseQuantizer<T>(scale) {}

        auto quantize(std::vector<Eigen::RowVector<T, 3>> const& points) const -> std::vector<Eigen::RowVector3i> {
            std::vector<Eigen::RowVector3i> quantized_points {};
            quantized_points.reserve(points.size());

            for (const auto& pt : points) {

                // Eigen::RowVector<T, 3> uvw = transform_point<T>(pt, get_inverse_transform_matrix<T>(TRUNC_OCT_SCALE_)) + Eigen::RowVector<T, 3>::Constant(0.5);
                Eigen::RowVector<T, 3> quantized_xyz = pt / (2 * this->TRUNC_OCT_SCALE_ / sqrt(5));
                Eigen::RowVector3i base = quantized_xyz.array().round().template cast<int>();

                if ((base[0] & 1 && base[1] & 1 && base[2] & 1) || (!(base[0] & 1) && !(base[1] & 1) && !(base[2] & 1))) {
                    quantized_points.push_back(base);
                    continue;
                }

                double min_dist = std::numeric_limits<double>::max();
                Eigen::RowVector3i best_voxel;

                for (const auto& offset : offsets) {
                    Eigen::RowVector3i candidate_voxel = base + offset;
                    Eigen::RowVector<T, 3> candidate_center = candidate_voxel.cast<T>() * 2 * this->TRUNC_OCT_SCALE_ / sqrt(5);
                    double dist = (pt - candidate_center).squaredNorm();

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_voxel = candidate_voxel;
                    }
                }

                quantized_points.push_back(best_voxel);
            }
            return quantized_points;
        }

        auto recover(std::vector<Eigen::RowVector3i> const& points) const -> std::vector<Eigen::RowVector<T, 3>> {
            std::vector<Eigen::RowVector<T, 3>> recovered_points {};
            recovered_points.reserve(points.size());

            for (const auto& voxel : points) {
                // recovered_points.push_back(transform_point<T>(voxel.cast<T>(), get_transform_matrix<T>(TRUNC_OCT_SCALE_)));
                recovered_points.push_back(voxel.cast<T>() * 2 * this->TRUNC_OCT_SCALE_ / sqrt(5));
            }

            return recovered_points;
        }

        private:

            const std::vector<Eigen::RowVector3i> offsets = {
                {0, 0, 1},
                {0, 0, -1},
                {0, 1, 0},
                {0, -1, 0},
                {1, 0, 0},
                {-1, 0, 0},
            };
    };

} // namespace TonSZ

#endif // TRUNCATED_OCTAHEDRON_QUANTIZER_HPP
