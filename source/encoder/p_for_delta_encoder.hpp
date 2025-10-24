#ifndef P_FOR_DELTA_ENCODER_HPP
#define P_FOR_DELTA_ENCODER_HPP

#include <cstddef>
#include <vector>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <algorithm>

namespace XnYZip {

    template<typename T>
    struct PForDeltaData {
        std::vector<T> patches;
        std::vector<T> deltas;

        PForDeltaData() = default;
        ~PForDeltaData() = default;
    };

    template<typename T>
    class PForDeltaEncoder {
        public:
            PForDeltaEncoder() = default;
            ~PForDeltaEncoder() = default;
            auto encode(const std::vector<T>& data) -> PForDeltaData<T> {
                PForDeltaData<T> result;
                result.deltas.reserve(data.size());
                result.patches.reserve(data.size() * 0.05);
            
                T lim = 254;
            
                for (const auto& value : data) {
                    if (value > lim) [[unlikely]] {
                        result.patches.push_back(value);
                        result.deltas.push_back(0);
                    } else {
                        result.deltas.push_back(value + 1);
                    }
                }
                return result;
            }

            auto decode(const PForDeltaData<T>& data) -> std::vector<T> {
                std::vector<T> result;
                result.reserve(data.deltas.size()); 
                size_t patch_idx = 0;
                for (const auto& delta_val : data.deltas) {
                    if (delta_val == 0) [[unlikely]] {
                        result.push_back(data.patches[patch_idx++]);
                    } else {
                        result.push_back(delta_val - 1);
                    }
                }
                return result;
            }
    };
} // namespace XnYZip

#endif // P_FOR_DELTA_ENCODER_HPP