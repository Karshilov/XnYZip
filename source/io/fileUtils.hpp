#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace XnYZip {
    template<typename T>
    auto read_file(const std::string& path) -> std::vector<Eigen::RowVector<T, 3>> {
        std::vector<Eigen::RowVector<T, 3>> points;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        T buffer[3];
        while (file.read(reinterpret_cast<char*>(buffer), sizeof(buffer))) {
            points.emplace_back(static_cast<T>(buffer[0]),
                                static_cast<T>(buffer[1]),
                                static_cast<T>(buffer[2]));
        }

        file.close();
        return points;
    }

    inline auto get_file_size(const std::string& path) -> std::uint64_t {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        auto pos = file.tellg();
        if (pos < 0) {
            throw std::runtime_error("Failed to query file size: " + path);
        }
        return static_cast<std::uint64_t>(pos);
    }

    template<typename T>
    auto read_file_chunk(const std::string& path,
                         std::uint64_t offset_bytes,
                         std::uint64_t count_bytes)
        -> std::vector<Eigen::RowVector<T, 3>> {
        constexpr std::uint64_t point_bytes = 3 * sizeof(T);
        if (offset_bytes % point_bytes != 0) {
            throw std::runtime_error("offset_bytes must be a multiple of " +
                                     std::to_string(point_bytes));
        }
        if (count_bytes % point_bytes != 0) {
            throw std::runtime_error("count_bytes must be a multiple of " +
                                     std::to_string(point_bytes));
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        file.seekg(0, std::ios::end);
        auto end_pos = file.tellg();
        if (end_pos < 0) {
            throw std::runtime_error("Failed to query file size: " + path);
        }
        auto file_size = static_cast<std::uint64_t>(end_pos);

        std::vector<Eigen::RowVector<T, 3>> points;
        if (offset_bytes >= file_size) {
            return points;
        }
        std::uint64_t remaining = file_size - offset_bytes;
        if (count_bytes == 0 || count_bytes > remaining) {
            count_bytes = remaining - (remaining % point_bytes);
        }

        file.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);

        std::uint64_t num_points = count_bytes / point_bytes;
        points.reserve(num_points);

        T buffer[3];
        for (std::uint64_t i = 0; i < num_points; ++i) {
            if (!file.read(reinterpret_cast<char*>(buffer), sizeof(buffer))) {
                throw std::runtime_error("read_file_chunk: short read at point " +
                                         std::to_string(i));
            }
            points.emplace_back(static_cast<T>(buffer[0]),
                                static_cast<T>(buffer[1]),
                                static_cast<T>(buffer[2]));
        }

        file.close();
        return points;
    }

    template<typename T>
    auto write_file(const std::string& path, const std::vector<Eigen::RowVector<T, 3>>& points) -> void {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        for (const auto& point : points) {
            file.write(reinterpret_cast<const char*>(point.data()), sizeof(point));
        }   

        file.close();
    }

    template<typename T>
    auto write_file_bin(const std::string& path, const std::vector<T>& data) -> void {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        for (const auto& value : data) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        file.close();
    }

    template<typename T>
    auto read_file_bin(const std::string& path) -> std::vector<T> {
        std::vector<T> data;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        T value;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            data.push_back(value);
        }

        file.close();
        return data;
    }
} // namespace XnYZip

#endif
