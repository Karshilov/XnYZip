#ifndef P_FOR_DELTA_ENCODER_HPP
#define P_FOR_DELTA_ENCODER_HPP

#include <cstddef>
#include <vector>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <algorithm>

namespace TonSZ {

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
                result.patches = std::vector<T>();
                result.deltas = std::vector<T>();

                auto copy_data = data;
                std::sort(copy_data.begin(), copy_data.end());
                T lim = 254;

                for (size_t i = 0; i < data.size(); i++) {
                    if (data[i] > lim) {
                        result.patches.push_back(data[i]);
                        result.deltas.push_back(0);
                    } else {
                        result.deltas.push_back(data[i] + 1);
                    }
                }
                return result;
            }

            auto decode(const PForDeltaData<T>& data) -> std::vector<T> {
                std::vector<T> result;
                for (size_t i = 0, idx = 0; i < data.deltas.size(); i++) {
                    if (data.deltas[i] == 0) {
                        result.push_back(data.patches[idx++]);
                    } else {
                        result.push_back(data.deltas[i] - 1);
                    }
                }
                return result;
            }
    };
} // namespace TonSZ

#endif // P_FOR_DELTA_ENCODER_HPP