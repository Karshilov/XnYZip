#ifndef BITPACKER_HPP
#define BITPACKER_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>

struct BitpackResult {
    std::vector<uint8_t> meta;
    std::vector<uint8_t> data;
};

template <typename T>
class BitPacker {
public:
    static auto pack(const std::vector<T>& values) -> BitpackResult {
        if (values.empty()) {
            throw std::invalid_argument("Input vector must not be empty");
        }
        uint64_t maxv = 0;
        for (auto v : values) {
            if (v > maxv) maxv = v;
        }
        uint8_t bitWidth = (maxv == 0 ? 1 : static_cast<uint8_t>(64 - __builtin_clzll(maxv)));
        
        BitpackResult out;
        uint32_t n = static_cast<uint32_t>(values.size());
        out.meta.reserve(5);
        for (int i = 0; i < 4; ++i) {
            out.meta.push_back(static_cast<uint8_t>((n >> (8*i)) & 0xFF));
        }
        out.meta.push_back(bitWidth);

        size_t totalBits = static_cast<size_t>(n) * bitWidth;
        size_t totalBytes = (totalBits + 7) / 8;
        out.data.assign(totalBytes, 0);

        size_t bitPos = 0;
        for (auto v : values) {
            for (uint8_t b = 0; b < bitWidth; ++b) {
                if (v & (uint64_t(1) << b)) {
                    size_t byteIdx = bitPos >> 3;
                    size_t bitIdx  = bitPos & 7;
                    out.data[byteIdx] |= static_cast<uint8_t>(1u << bitIdx);
                }
                ++bitPos;
            }
        }
        return out;
    }

    static auto unpack(const BitpackResult& in) -> std::vector<T> {
        if (in.meta.size() != 5) {
            throw std::runtime_error("Invalid meta size in bitpacked data");
        }

        uint32_t n = 0;
        for (int i = 0; i < 4; ++i) {
            n |= (uint32_t(in.meta[i]) << (8 * i));
        }

        uint8_t bitWidth = in.meta[4];
        std::vector<T> values(n, 0);

        size_t bitPos = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t val = 0;
            for (uint8_t b = 0; b < bitWidth; ++b) {
                size_t byteIdx = bitPos >> 3;
                size_t bitIdx  = bitPos & 7;
                if (in.data[byteIdx] & (1u << bitIdx)) {
                    val |= (uint64_t(1) << b);
                }
                ++bitPos;
            }
            values[i] = static_cast<T>(val);
        }
        return values;
    }
};


#endif // BITPACKER_HPP
