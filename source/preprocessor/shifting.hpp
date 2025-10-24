#ifndef SHIFTING_HPP
#define SHIFTING_HPP

#include <vector>
#include <Eigen/Dense>
#include <limits>

namespace XnYZip {

    template<typename T>
    auto shift_points(std::vector<Eigen::RowVector<T, 3>> const& points, Eigen::RowVector<T, 3> & offset) -> std::vector<Eigen::RowVector<T, 3>> {
        if (points.empty()) return {};

        T min_x = std::numeric_limits<T>::max();
        T min_y = std::numeric_limits<T>::max();
        T min_z = std::numeric_limits<T>::max();

        for (const auto& pt : points) {
            if (pt[0] < min_x) min_x = pt[0];
            if (pt[1] < min_y) min_y = pt[1];
            if (pt[2] < min_z) min_z = pt[2];
        }

        offset[0] = min_x;
        offset[1] = min_y;
        offset[2] = min_z;

        std::vector<Eigen::RowVector<T, 3>> shifted_points;
        shifted_points.reserve(points.size());

        for (const auto& pt : points) {
            shifted_points.push_back(pt - offset);
        }

        return shifted_points;
    }

    template<typename T>
    auto unshift_points(std::vector<Eigen::RowVector<T, 3>> const& points, Eigen::RowVector<T, 3> const& offset) -> std::vector<Eigen::RowVector<T, 3>> {
        if (points.empty()) return {};

        std::vector<Eigen::RowVector<T, 3>> unshifted_points;
        unshifted_points.reserve(points.size());

        for (const auto& pt : points) {
            unshifted_points.push_back(pt + offset);
        }

        return unshifted_points;
    }

}  // namespace XnYZip

#endif  // SHIFTING_HPP
