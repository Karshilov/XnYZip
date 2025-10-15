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

namespace XnYSZ {

// A simple Huffman encoder/decoder using a codebook lookup.
// Supports integral symbol types up to 64 bits.

template<typename T>
class HuffmanEncoder {
    static_assert(std::is_integral<T>::value, "Symbol type must be integral");

    struct Node {
        T symbol;
        uint64_t freq;
        std::shared_ptr<Node> left, right;
        bool isLeaf() const { return !left && !right; }
    };

public:
    void build(const std::vector<T>& data) {
        std::unordered_map<T, uint64_t> freq;
        for (auto v : data) freq[v]++;

        double entropy = 0.0;
        for (auto& kv : freq) {
            double p = double(kv.second) / data.size();
            entropy -= p * std::log2(p);
        }
        std::cout << "entropy: " << entropy << std::endl;

        auto cmp = [](auto const &a, auto const &b){ return a->freq > b->freq; };
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, decltype(cmp)> pq(cmp);

        for (auto& kv : freq) {
            auto node = std::make_shared<Node>();
            node->symbol = kv.first;
            node->freq = kv.second;
            pq.push(node);
        }
        while (pq.size() > 1) {
            auto a = pq.top(); pq.pop();
            auto b = pq.top(); pq.pop();
            auto p = std::make_shared<Node>();
            p->freq = a->freq + b->freq;
            p->left = a; p->right = b;
            pq.push(p);
        }
        auto root = pq.top();
        code_table.clear();
        build_code(root, 0u, 0u);
        build_decode_map();
    }

    std::vector<uint8_t> get_meta() const {
        std::vector<uint8_t> out;
        uint32_t count = uint32_t(code_table.size());
        append_uint32(out, count);
        for (auto& kv : code_table) {
            append_symbol(out, kv.first);
            out.push_back(kv.second.length);
            append_uint32(out, kv.second.bits);
        }
        return out;
    }

    size_t meta_size() const { return get_meta().size(); }

    std::vector<uint8_t> encode(const std::vector<T>& data) const {
        std::vector<uint8_t> out;
        uint64_t bitbuf = 0;
        int bits = 0;
        for (auto v : data) {
            auto const &c = code_table.at(v);
            bitbuf = (bitbuf << c.length) | c.bits;
            bits += c.length;
            while (bits >= 8) {
                out.push_back(uint8_t(bitbuf >> (bits - 8)));
                bits -= 8;
                bitbuf &= ((1ULL << bits) - 1);
            }
        }
        if (bits) out.push_back(uint8_t(bitbuf << (8 - bits)));
        return out;
    }

    void load_meta(const std::vector<uint8_t>& meta) {
        code_table.clear();
        size_t pos = 0;
        uint32_t count = read_uint32(meta, pos);
        for (uint32_t i = 0; i < count; ++i) {
            T sym = read_symbol(meta, pos);
            uint8_t len = meta[pos++];
            uint32_t bitsv = read_uint32(meta, pos);
            code_table[sym] = {bitsv, len};
        }
        build_decode_map();
    }

    std::vector<T> decode(const std::vector<uint8_t>& data, size_t orig_size) const {
        std::vector<T> out;
        out.reserve(orig_size);
    
        if (code_table.size() == 1) {
            auto sym = code_table.begin()->first;
            out.assign(orig_size, sym);
            return out;
        }
    
        uint32_t buf = 0;
        uint8_t buflen = 0;
    
        auto emit = [&]() {
            auto itl = decode_map.find(buflen);
            if (itl != decode_map.end()) {
                auto its = itl->second.find(buf);
                if (its != itl->second.end()) {
                    out.push_back(its->second);
                    buf = 0;
                    buflen = 0;
                }
            }
        };
    
        for (size_t i = 0; i < data.size() * 8 && out.size() < orig_size; ++i) {
            size_t idx = i >> 3;
            int shift = 7 - (i & 7);
            buf = (buf << 1) | ((data[idx] >> shift) & 1);
            ++buflen;
            emit();
        }
    
        return out;
    }
    
private:
    struct Code { uint32_t bits; uint8_t length; };
    std::unordered_map<T, Code> code_table;
    std::unordered_map<uint8_t, std::unordered_map<uint32_t, T>> decode_map;

    void build_decode_map() {
        decode_map.clear();
        for (auto& kv : code_table)
            decode_map[kv.second.length][kv.second.bits] = kv.first;
    }

    void build_code(std::shared_ptr<Node> const &node, uint32_t prefix, uint8_t len) {
        if (node->isLeaf()) {
            if (len == 0) {
                code_table[node->symbol] = {0, 1};
            } else {
                code_table[node->symbol] = {prefix, len};
            }
            return;
        }
    
        build_code(node->left, prefix << 1, len + 1);
        build_code(node->right, (prefix << 1) | 1u, len + 1);
    }

    static void append_uint32(std::vector<uint8_t>& o, uint32_t v) {
        o.push_back(v>>24); o.push_back(v>>16);
        o.push_back(v>>8);  o.push_back(v);
    }
    static uint32_t read_uint32(std::vector<uint8_t> const &i, size_t &p) {
        uint32_t v = (uint32_t(i[p])<<24)|(uint32_t(i[p+1])<<16)|
                     (uint32_t(i[p+2])<<8)|uint32_t(i[p+3]); p+=4; return v;
    }

    static void append_uint64(std::vector<uint8_t>& o, uint64_t v) {
        for (int shift=56; shift>=0; shift-=8) o.push_back(uint8_t(v>>shift));
    }
    static uint64_t read_uint64(std::vector<uint8_t> const &i, size_t &p) {
        uint64_t v=0;
        for (int k=0; k<8; ++k) v=(v<<8)|i[p++];
        return v;
    }

    static void append_symbol(std::vector<uint8_t>& o, T sym) {
        if constexpr(sizeof(T)==8) append_uint64(o, uint64_t(sym));
        else o.insert(o.end(), reinterpret_cast<uint8_t const*>(&sym),
                      reinterpret_cast<uint8_t const*>(&sym)+sizeof(T));
    }
    static T read_symbol(std::vector<uint8_t> const &i, size_t &p) {
        if constexpr(sizeof(T)==8) return T(read_uint64(i,p));
        T v; std::memcpy(&v, i.data()+p, sizeof(T)); p+=sizeof(T); return v;
    }
};

} // namespace XnYSZ

#endif // HUFFMAN_ENCODER_HPP