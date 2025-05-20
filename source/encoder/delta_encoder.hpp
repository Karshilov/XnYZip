#ifndef DELTA_ENCODER_HPP
#define DELTA_ENCODER_HPP

#include <vector>
#include <Eigen/Dense>

namespace TonSZ {

    template<typename T>
    class DeltaEncoder {
    public:
        DeltaEncoder() {}

        auto encode(const std::vector<Eigen::RowVector<T, 3>>& points) -> std::vector<Eigen::RowVector<T, 3>> {
            if (points.empty()) return {};

            std::vector<Eigen::RowVector<T, 3>> deltas;
            deltas.reserve(points.size());

            deltas.push_back(points[0]);  
            for (size_t i = 1; i < points.size(); ++i) {
                deltas.push_back(points[i] - points[i - 1]);
            }

            return deltas;
        }

        auto decode(const std::vector<Eigen::RowVector<T, 3>>& deltas) -> std::vector<Eigen::RowVector<T, 3>> {
            if (deltas.empty()) return {};

            std::vector<Eigen::RowVector<T, 3>> points;
            points.reserve(deltas.size());

            points.push_back(deltas[0]);
            for (size_t i = 1; i < deltas.size(); ++i) {
                points.push_back(points[i - 1] + deltas[i]);
            }

            return points;
        }
    };

}  // namespace TonSZ

#endif  // DELTA_ENCODER_HPP
