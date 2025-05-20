#ifndef QUANTIZER_UTILS_HPP
#define QUANTIZER_UTILS_HPP

#include <Eigen/Dense>
#include <cmath>

namespace TonSZ {

    template<typename T>
    inline auto get_transform_matrix(float const TRUNC_OCT_SCALE) -> Eigen::Matrix<T, 3, 3> {
        Eigen::Matrix<T, 3, 3> A;
        A << 1, 1, 0,
             1, 0, 1,
             0, 1, 1;
        return TRUNC_OCT_SCALE * A;
    }

    template<typename T>
    inline auto get_inverse_transform_matrix(float const TRUNC_OCT_SCALE) -> Eigen::Matrix<T, 3, 3> {
        return get_transform_matrix<T>(TRUNC_OCT_SCALE).inverse();
    }

    template<typename T>
    inline auto transform_point(Eigen::RowVector<T, 3> const& point, Eigen::Matrix<T, 3, 3> const& transform_matrix) -> Eigen::RowVector<T, 3> {
        return point * transform_matrix.transpose();
    }

    template<typename T>
    inline auto inverse_transform_point(Eigen::RowVector<T, 3> const& point, Eigen::Matrix<T, 3, 3> const& inverse_transform_matrix) -> Eigen::RowVector<T, 3> {
        return point * inverse_transform_matrix.transpose();
    }

} // namespace TonSZ

#endif // QUANTIZER_UTILS_HPP