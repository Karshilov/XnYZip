#ifndef QUANTIZER_UTILS_HPP
#define QUANTIZER_UTILS_HPP

#include <Eigen/Dense>
#include <cmath>

namespace TonSZ {

    template<typename T>
    inline auto get_transform_matrix(float const TRUNC_OCT_SCALE) -> Eigen::Matrix<T, 3, 3> {
        Eigen::Matrix<T, 3, 3> A;
        A << 0, 1, 1,
             1, 0, 1,
             1, 1, 0;
        return A;
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
        printf("point: %f %f %f\n", point[0], point[1], point[2]);
        printf("inverse_transform_matrix: %f %f %f\n", inverse_transform_matrix(0, 0), inverse_transform_matrix(0, 1), inverse_transform_matrix(0, 2));
        printf("inverse_transform_matrix: %f %f %f\n", inverse_transform_matrix(1, 0), inverse_transform_matrix(1, 1), inverse_transform_matrix(1, 2));
        printf("inverse_transform_matrix: %f %f %f\n", inverse_transform_matrix(2, 0), inverse_transform_matrix(2, 1), inverse_transform_matrix(2, 2));
        auto result = point * inverse_transform_matrix.transpose();
        printf("result: %f %f %f\n", result[0], result[1], result[2]);
        return result;
    }

} // namespace TonSZ

#endif // QUANTIZER_UTILS_HPP