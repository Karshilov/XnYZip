#ifndef TON_SZ_BLOCK_ENCODING_PROFILE_HPP
#define TON_SZ_BLOCK_ENCODING_PROFILE_HPP

#include <cstdint>
#include <vector>

namespace XnYZip {

// Shared encoding parameters that can be broadcast across MPI ranks so every
// rank emits a bit-identical encoding of the dominant entropy-coded streams
// (repos and quads), enabling zstd to find cross-chunk redundancy in the
// final compressed parts.
//
// The single-machine compress() path is unaware of this struct; only the new
// compress_with_profile() / fit_profile() methods consume it. See
// BlockCompressor for usage.
struct BlockEncodingProfile {
    // Serialized Huffman metadata (output of HuffmanEncoder<uint64_t>::get_meta())
    // for the per-point repos delta stream after PFD.
    std::vector<uint8_t> repos_huff_meta;

    // Serialized Huffman metadata (output of HuffmanEncoder<uint8_t>::get_meta())
    // for the per-point quads delta stream after PFD.
    std::vector<uint8_t> quads_huff_meta;

    // Returns true if the profile carries non-empty metas for both streams.
    auto valid() const -> bool {
        return !repos_huff_meta.empty() && !quads_huff_meta.empty();
    }
};

// Simple flat serialization for MPI broadcast:
//   [u32 repos_meta_size][repos_meta_bytes][u32 quads_meta_size][quads_meta_bytes]
inline auto serialize_profile(const BlockEncodingProfile& p) -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    auto append_u32 = [&](std::uint32_t v) {
        out.push_back(static_cast<std::uint8_t>(v & 0xFF));
        out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFF));
    };
    append_u32(static_cast<std::uint32_t>(p.repos_huff_meta.size()));
    out.insert(out.end(), p.repos_huff_meta.begin(), p.repos_huff_meta.end());
    append_u32(static_cast<std::uint32_t>(p.quads_huff_meta.size()));
    out.insert(out.end(), p.quads_huff_meta.begin(), p.quads_huff_meta.end());
    return out;
}

inline auto deserialize_profile(const std::vector<std::uint8_t>& blob) -> BlockEncodingProfile {
    BlockEncodingProfile p;
    auto read_u32 = [&](std::size_t& pos) -> std::uint32_t {
        std::uint32_t v = static_cast<std::uint32_t>(blob[pos]) |
                          (static_cast<std::uint32_t>(blob[pos + 1]) << 8) |
                          (static_cast<std::uint32_t>(blob[pos + 2]) << 16) |
                          (static_cast<std::uint32_t>(blob[pos + 3]) << 24);
        pos += 4;
        return v;
    };
    std::size_t pos = 0;
    if (blob.size() < 4) return p;
    auto repos_n = read_u32(pos);
    if (pos + repos_n > blob.size()) return p;
    p.repos_huff_meta.assign(blob.begin() + pos, blob.begin() + pos + repos_n);
    pos += repos_n;
    if (pos + 4 > blob.size()) return p;
    auto quads_n = read_u32(pos);
    if (pos + quads_n > blob.size()) return p;
    p.quads_huff_meta.assign(blob.begin() + pos, blob.begin() + pos + quads_n);
    return p;
}

} // namespace XnYZip

#endif // TON_SZ_BLOCK_ENCODING_PROFILE_HPP
