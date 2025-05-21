#ifndef TON_SZ_UTILS_HPP
#define TON_SZ_UTILS_HPP

#include <vector>
#include <string>

namespace TonSZ {

    enum class QUANTIZER_TYPE {
        CUBE,
        TRUNCATED_OCTAHEDRON,
    };

    inline auto split_string(const std::string& input, const std::string& delimiter) -> std::vector<std::string> {
        std::vector<std::string> result;
        size_t start = 0;
        size_t end;
        while ((end = input.find(delimiter, start)) != std::string::npos) {
            result.push_back(input.substr(start, end - start));
            start = end + delimiter.length();
        }
        result.push_back(input.substr(start)); // Add last part
        return result;
    }

}
#endif // TON_SZ_UTILS_HPP
