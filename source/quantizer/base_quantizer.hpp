#ifndef TON_SZ_BASE_QUANTIZER_HPP
#define TON_SZ_BASE_QUANTIZER_HPP

#include <Eigen/Dense>
#include <vector>

namespace TonSZ {

    template<typename T>
    class BaseQuantizer {

        public:

            BaseQuantizer(T const TRUNC_OCT_SCALE) : TRUNC_OCT_SCALE_(TRUNC_OCT_SCALE) {}

            virtual ~BaseQuantizer() = default;

            virtual auto quantize(std::vector<Eigen::RowVector<T, 3>> const& points) const -> std::vector<Eigen::RowVector3i> = 0;

            virtual auto recover(std::vector<Eigen::RowVector3i> const& quantized_points) const -> std::vector<Eigen::RowVector<T, 3>> = 0;
        protected:
            T const TRUNC_OCT_SCALE_;
    };

}

#endif
