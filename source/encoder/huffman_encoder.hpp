#ifndef HUFFMAN_ENCODER_HPP
#define HUFFMAN_ENCODER_HPP

#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <cmath>

namespace TonSZ {

// A simple Huffman encoder/decoder using a codebook lookup for decoding.
// Assumes symbol type is integral and code length does not exceed 32 bits.

template<typename T>
class HuffmanEncoder {
    static_assert(std::is_integral<T>::value, "Symbol type must be integral");

    // internal tree node structure
    struct Node {
        T symbol;
        uint64_t freq;
        std::shared_ptr<Node> left, right;
        bool isLeaf() const { return !left && !right; }
    };

public:
    // Build Huffman code table from input data
    void build(const std::vector<T>& data) {
        std::unordered_map<T, uint64_t> freq;
        for (auto v : data) freq[v]++;

        // calc entropy
        double entropy = 0.0;
        for (auto& kv : freq) {
            double p = static_cast<double>(kv.second) / data.size();
            entropy -= p * std::log2(p);
        }
        std::cout << "entropy: " << entropy << std::endl;

        auto cmp = [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
            return a->freq > b->freq;
        };
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, decltype(cmp)> pq(cmp);

        // push leaves
        for (auto& kv : freq) {
            auto node = std::make_shared<Node>();
            node->symbol = kv.first;
            node->freq = kv.second;
            pq.push(node);
        }
        // build tree
        while (pq.size() > 1) {
            auto a = pq.top(); pq.pop();
            auto b = pq.top(); pq.pop();
            auto parent = std::make_shared<Node>();
            parent->freq = a->freq + b->freq;
            parent->left = a;
            parent->right = b;
            pq.push(parent);
        }
        auto root = pq.top();

        // build code table and decode map
        code_table.clear();
        build_code(root, 0u, 0u);
        build_decode_map();
    }

    // Serialize code table to a byte vector (meta info)
    std::vector<uint8_t> get_meta() const {
        std::vector<uint8_t> out;
        uint32_t count = static_cast<uint32_t>(code_table.size());
        append_uint32(out, count);
        for (auto& kv : code_table) {
            T sym = kv.first;
            auto c = kv.second;
            append_bytes(out, &sym, sizeof(T));
            out.push_back(c.length);
            append_uint32(out, c.bits);
        }
        return out;
    }

    // Return size of serialized meta info
    size_t meta_size() const { return get_meta().size(); }

    // Encode data into tightly packed bits (byte-at-a-time flush)
    std::vector<uint8_t> encode(const std::vector<T>& data) const {
        std::vector<uint8_t> out;
        uint64_t bit_buffer = 0;
        int bit_count = 0;
        for (auto v : data) {
            const auto& c = code_table.at(v);
            bit_buffer = (bit_buffer << c.length) | c.bits;
            bit_count += c.length;
            while (bit_count >= 8) {
                uint8_t byte = static_cast<uint8_t>(bit_buffer >> (bit_count - 8));
                out.push_back(byte);
                bit_count -= 8;
                bit_buffer &= ((1ULL << bit_count) - 1);
            }
        }
        if (bit_count > 0) {
            uint8_t byte = static_cast<uint8_t>(bit_buffer << (8 - bit_count));
            out.push_back(byte);
        }
        return out;
    }

    // Load code table from serialized meta info and prepare decode map
    void load_meta(const std::vector<uint8_t>& meta) {
        code_table.clear();
        size_t pos = 0;
        uint32_t count = read_uint32(meta, pos);
        for (uint32_t i = 0; i < count; ++i) {
            T sym = read_value<T>(meta, pos);
            uint8_t len = meta[pos++];
            uint32_t bits = read_uint32(meta, pos);
            code_table[sym] = {bits, len};
        }
        build_decode_map();
    }

    // Decode packed data using codebook lookup
    std::vector<T> decode(const std::vector<uint8_t>& data, size_t original_size) const {
        std::vector<T> out;
        out.reserve(original_size);

        uint32_t code_buffer = 0;
        uint8_t buffer_len = 0;
        auto emit_if_match = [&](void) {
            auto it_len = decode_map.find(buffer_len);
            if (it_len != decode_map.end()) {
                auto it_sym = it_len->second.find(code_buffer);
                if (it_sym != it_len->second.end()) {
                    out.push_back(it_sym->second);
                    code_buffer = 0;
                    buffer_len = 0;
                }
            }
        };

        for (size_t i = 0; i < data.size() * 8 && out.size() < original_size; ++i) {
            size_t byteIdx = i >> 3;
            int shift = 7 - (i & 7);
            bool bit = (data[byteIdx] >> shift) & 1u;
            code_buffer = (code_buffer << 1) | static_cast<uint32_t>(bit);
            buffer_len++;
            emit_if_match();
        }
        return out;
    }

private:
    struct Code { uint32_t bits; uint8_t length; };
    std::unordered_map<T, Code> code_table;
    std::unordered_map<uint8_t, std::unordered_map<uint32_t, T>> decode_map;

    // Build decode lookup from code_table
    void build_decode_map() {
        decode_map.clear();
        for (auto& kv : code_table) {
            decode_map[kv.second.length][kv.second.bits] = kv.first;
        }
    }

    // Recursive helper to build code table
    void build_code(const std::shared_ptr<Node>& node, uint32_t prefix, uint8_t length) {
        if (node->isLeaf()) {
            code_table[node->symbol] = {prefix, length};
            return;
        }
        build_code(node->left, prefix << 1, length + 1);
        build_code(node->right, (prefix << 1) | 1u, length + 1);
    }

    // utility: append 32-bit BE
    static void append_uint32(std::vector<uint8_t>& out, uint32_t v) {
        out.push_back(static_cast<uint8_t>(v >> 24));
        out.push_back(static_cast<uint8_t>(v >> 16));
        out.push_back(static_cast<uint8_t>(v >> 8));
        out.push_back(static_cast<uint8_t>(v));
    }
    static void append_bytes(std::vector<uint8_t>& out, const void* ptr, size_t n) {
        auto p = reinterpret_cast<const uint8_t*>(ptr);
        out.insert(out.end(), p, p + n);
    }
    static uint32_t read_uint32(const std::vector<uint8_t>& in, size_t& pos) {
        uint32_t v = (static_cast<uint32_t>(in[pos]) << 24) |
                     (static_cast<uint32_t>(in[pos+1]) << 16) |
                     (static_cast<uint32_t>(in[pos+2]) << 8) |
                     static_cast<uint32_t>(in[pos+3]);
        pos += 4;
        return v;
    }
    template<typename U>
    static U read_value(const std::vector<uint8_t>& in, size_t& pos) {
        U val;
        std::memcpy(&val, in.data() + pos, sizeof(U));
        pos += sizeof(U);
        return val;
    }
};

} // namespace huffman

#endif // HUFFMAN_ENCODER_HPP
