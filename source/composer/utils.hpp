#ifndef TON_SZ_UTILS_HPP
#define TON_SZ_UTILS_HPP

#include <vector>
#include <string>
#include <zstd.h>
#include <stdexcept>

namespace XnYSZ {

    inline auto decompress_buffer(const std::vector<uint8_t>& compressed) -> std::vector<uint8_t> {
        unsigned long long const frame_size =
            ZSTD_getFrameContentSize(compressed.data(), compressed.size());
        if (frame_size == ZSTD_CONTENTSIZE_ERROR) {
            throw std::runtime_error("Not a zstd frame");
        }
        if (frame_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Original size unknown");
        }
        std::vector<uint8_t> out(frame_size);
        size_t const d_size = ZSTD_decompress(
            out.data(), frame_size,
            compressed.data(), compressed.size()
        );
        if (ZSTD_isError(d_size)) {
            throw std::runtime_error(ZSTD_getErrorName(d_size));
        }
        if (d_size != frame_size) {
            throw std::runtime_error("Decompressed size mismatch");
        }
        return out;
    }


    auto compress_u8_vector(const std::vector<uint8_t>& data, int compression_level = 3) -> std::vector<uint8_t> {
        size_t input_size = data.size() * sizeof(uint8_t);
        size_t max_compressed_size = ZSTD_compressBound(input_size);

        std::vector<uint8_t> compressed(max_compressed_size);

        size_t compressed_size = ZSTD_compress(
            compressed.data(), max_compressed_size,
            data.data(), input_size,
            compression_level
        );

        if (ZSTD_isError(compressed_size)) {
            throw std::runtime_error(ZSTD_getErrorName(compressed_size));
        }

        compressed.resize(compressed_size);  // Trim unused space
        return compressed;
    }

    enum class QUANTIZER_TYPE {
        CUBE,
        TRUNCATED_OCTAHEDRON,
        ADAPTIVE,
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
