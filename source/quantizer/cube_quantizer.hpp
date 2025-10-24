#ifndef CUBE_QUANTIZER_HPP
#define CUBE_QUANTIZER_HPP

#include "base_quantizer.hpp"
#include <vector>
#include <Eigen/Dense>

namespace XnYZip {

    template<typename T>
    class CubeQuantizer : public BaseQuantizer<T> {
        public:
            explicit CubeQuantizer(T scale): BaseQuantizer<T>(scale) {}

        auto quantize(std::vector<Eigen::RowVector<T, 3>> const& points, std::vector<T>& params) const -> std::vector<Eigen::RowVector3i> {
            std::vector<Eigen::RowVector3i> quantized_points {};
            quantized_points.reserve(points.size());

            for (const auto& pt : points) {
                Eigen::RowVector<T, 3> quantized_xyz = pt / (2 * this->TRUNC_OCT_SCALE_ / sqrt(3));
                Eigen::RowVector3i base = quantized_xyz.array().floor().template cast<int>();

                quantized_points.push_back(base);
            }

            return quantized_points;
        }

        auto recover(std::vector<Eigen::RowVector3i> const& points, std::vector<T> const& params) const -> std::vector<Eigen::RowVector<T, 3>> {
            std::vector<Eigen::RowVector<T, 3>> recovered_points {};
            recovered_points.reserve(points.size());

            for (const auto& voxel : points) {
                recovered_points.push_back((voxel.cast<T>() + Eigen::RowVector<T, 3>::Constant(0.5)) * 2 * this->TRUNC_OCT_SCALE_ / sqrt(3));
            }

            return recovered_points;
        }

    };

} // namespace XnYZip

#endif // CUBE_QUANTIZER_HPP
