#ifndef GOLOMBRICECODER_HPP
#define GOLOMBRICECODER_HPP

#include <vector>
#include <cstdint>
#include <stdexcept> 
#include <cmath>     
#include <limits>    
#include "../encoder/bitpacking_encoder.hpp"


class BitWriter {
    std::vector<uint8_t>& buffer_;
    size_t bit_pos_ = 0;

public:
    explicit BitWriter(std::vector<uint8_t>& buffer) : buffer_(buffer) {}

    void write_bit(bool bit) {
        size_t byte_idx = bit_pos_ / 8;
        size_t bit_idx_in_byte = bit_pos_ % 8;
        if (byte_idx >= buffer_.size()) {
            buffer_.push_back(0);
        }
        if (bit) {
            buffer_[byte_idx] |= (1u << bit_idx_in_byte);
        }
        bit_pos_++;
    }

    void write_unary(uint64_t q) {
        for (uint64_t i = 0; i < q; ++i) {
            write_bit(true);
        }
        write_bit(false);
    }

    void write_k_bits(uint64_t val, uint8_t k) {
        for (uint8_t b = 0; b < k; ++b) {
            write_bit((val >> b) & 1);
        }
    }

    auto total_bits_written() const -> size_t {
        return bit_pos_;
    }
};

class BitReader {
    const std::vector<uint8_t>& buffer_;
    size_t bit_pos_ = 0;
    const size_t total_bits_available_;

public:
    explicit BitReader(const std::vector<uint8_t>& buffer)
        : buffer_(buffer), total_bits_available_(buffer.size() * 8) {}

    auto read_bit() -> bool {
        if (bit_pos_ >= total_bits_available_) {
            throw std::out_of_range("Attempting to read beyond bitstream boundary");
        }
        size_t byte_idx = bit_pos_ / 8;
        size_t bit_idx_in_byte = bit_pos_ % 8;
        bool bit = ((buffer_[byte_idx] >> bit_idx_in_byte) & 1) != 0;
        bit_pos_++;
        return bit;
    }

    auto read_unary() -> uint64_t {
        uint64_t q = 0;
        while (read_bit()) {
            q++;
        }
        return q;
    }

    uint64_t read_k_bits(uint8_t k) {
        uint64_t val = 0;
        for (uint8_t b = 0; b < k; ++b) {
            if (read_bit()) {
                val |= (1ULL << b);
            }
        }
        return val;
    }
};


template <typename T>
class GolombRiceCoder {
public:
    static auto pack(const std::vector<T>& values) -> BitpackResult {
        if (values.empty()) {
            return BitpackResult{};
        }

        uint64_t sum_of_values = 0;
        bool all_zeros = true;
        for (const T& val_t : values) {
            if (val_t < 0) {
                throw std::invalid_argument("Golomb-Rice codes are defined for non-negative integers only.");
            }
            if (val_t != 0) {
                all_zeros = false;
            }
            sum_of_values += static_cast<uint64_t>(val_t);
        }

        uint8_t k_param;
        if (all_zeros) {
            k_param = 0;
        } else {
            double average = static_cast<double>(sum_of_values) / values.size();
            if (average < 1.0) { 
                k_param = 0;
            } else {
                k_param = static_cast<uint8_t>(std::floor(std::log2(average)));
            }
        }
        
        if (k_param >= 63) {
            k_param = 62;
        }


        BitpackResult out;
        uint32_t n = static_cast<uint32_t>(values.size());
        
        out.meta.resize(5);
        for (int i = 0; i < 4; ++i) {
            out.meta[i] = static_cast<uint8_t>((n >> (8 * i)) & 0xFF);
        }
        out.meta[4] = k_param;

        BitWriter writer(out.data);
        uint64_t M = 1ULL << k_param;

        for (const T& val_t : values) {
            uint64_t v = static_cast<uint64_t>(val_t);
            uint64_t q = v / M;
            uint64_t r = v % M;
            
            writer.write_unary(q);
            writer.write_k_bits(r, k_param);
        }
        
        return out;
    }

    static auto unpack(const BitpackResult& in) -> std::vector<T> {
        if (in.meta.empty() && in.data.empty()) {
            return std::vector<T>{};
        }

        if (in.meta.size() != 5) {
            throw std::runtime_error("Invalid meta size in Golomb-Rice packed data. Expected 5 bytes.");
        }

        uint32_t n = 0;
        for (int i = 0; i < 4; ++i) {
            n |= (static_cast<uint32_t>(in.meta[i]) << (8 * i));
        }

        if (n == 0) {
             return std::vector<T>{};
        }

        uint8_t k_param = in.meta[4];
        if (k_param >= 63 ) {
             throw std::runtime_error("Invalid k_param in Golomb-Rice packed data meta (k too large).");
        }


        std::vector<T> values;
        values.reserve(n);

        BitReader reader(in.data);
        uint64_t M = 1ULL << k_param;

        for (uint32_t i = 0; i < n; ++i) {
            uint64_t q = reader.read_unary();
            uint64_t r = reader.read_k_bits(k_param);
            uint64_t reconstructed_val = q * M + r;

            if (reconstructed_val > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
                throw std::runtime_error("Unpacked value exceeds capacity of target type T.");
            }
            values.push_back(static_cast<T>(reconstructed_val));
        }
        
        return values;
    }
};

#endif // GOLOMBRICECODER_HPP