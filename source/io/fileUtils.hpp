#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace TonSZ {
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
} // namespace TonSZ

#endif
