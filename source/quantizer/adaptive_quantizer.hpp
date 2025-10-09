#ifndef ADAPTIVE_QUANTIZER_HPP
#define ADAPTIVE_QUANTIZER_HPP

#include "base_quantizer.hpp"
#include <vector>
#include <Eigen/Dense>
#include <cmath>

namespace XnYSZ {

    template<typename T>
    class AdaptiveQuantizer : public BaseQuantizer<T> {
    public:
        explicit AdaptiveQuantizer(T scale): BaseQuantizer<T>(scale) {}

        auto quantize(std::vector<Eigen::RowVector<T, 3>> const& points, std::vector<T>& params) const -> std::vector<Eigen::RowVector3i> {
            size_t n = points.size();
            // T sum_x = 0;
            // T sum_y = 0;
            // T sum_z = 0;
            // T sum_x2 = 0;
            // T sum_y2 = 0;
            // T sum_z2 = 0;
            // for (auto const& pt : points) {
            //     sum_x  += pt.x(); sum_y  += pt.y(); sum_z  += pt.z();
            //     sum_x2 += pt.x() * pt.x();
            //     sum_y2 += pt.y() * pt.y();
            //     sum_z2 += pt.z() * pt.z();
            // }
            // T sigma_x = 0;
            // T sigma_y = 0;
            // T sigma_z = 0;
            // if (n > 0) {
            //     T mx = sum_x / n;
            //     T my = sum_y / n;
            //     T mz = sum_z / n;
            //     sigma_x = std::sqrt(std::max(sum_x2 / n - mx * mx, T(0)));
            //     sigma_y = std::sqrt(std::max(sum_y2 / n - my * my, T(0)));
            //     sigma_z = std::sqrt(std::max(sum_z2 / n - mz * mz, T(0)));
            // }
            T min_x = std::numeric_limits<T>::max();
            T min_y = std::numeric_limits<T>::max();
            T min_z = std::numeric_limits<T>::max();
            T max_x = std::numeric_limits<T>::min();
            T max_y = std::numeric_limits<T>::min();
            T max_z = std::numeric_limits<T>::min();
            for (auto const& pt : points) {
                min_x = std::min(min_x, pt.x());
                min_y = std::min(min_y, pt.y());
                min_z = std::min(min_z, pt.z());
                max_x = std::max(max_x, pt.x());
                max_y = std::max(max_y, pt.y());
                max_z = std::max(max_z, pt.z());
            }
            T R = this->TRUNC_OCT_SCALE_;
            T norm = std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) + (max_z - min_z) * (max_z - min_z));
            T a;
            T b;
            T c;
            if (norm > 0) {
                a = R * (max_x - min_x) / norm;
                b = R * (max_y - min_y) / norm;
                c = R * (max_z - min_z) / norm;
            } else {
                a = b = c = R / std::sqrt(T(3));
            }

            std::vector<Eigen::RowVector3i> quantized;
            quantized.reserve(n);
            for (auto const& pt : points) {
                Eigen::RowVector<T,3> v;
                v.x() = pt.x() / a;
                v.y() = pt.y() / b;
                v.z() = pt.z() / c;
                quantized.push_back(v.array().round().template cast<int>());
            }
            params = {a, b, c};
            printf("params: %f, %f, %f\n", a, b, c);
            return quantized;
        }

        auto recover(std::vector<Eigen::RowVector3i> const& points, std::vector<T> const& params) const -> std::vector<Eigen::RowVector<T, 3>> {
            T a = params[0];
            T b = params[1];
            T c = params[2];

            std::vector<Eigen::RowVector<T,3>> recovered;
            recovered.reserve(points.size());
            for (auto const& idx : points) {
                Eigen::RowVector<T,3> pt;
                pt.x() = idx.x() * a;
                pt.y() = idx.y() * b;
                pt.z() = idx.z() * c;
                recovered.push_back(pt);
            }
            return recovered;
        }
    };

} // namespace XnYSZ

#endif // ADAPTIVE_QUANTIZER_HPP
