#ifndef Z_ORDER_CURVE_HPP
#define Z_ORDER_CURVE_HPP

#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace TonSZ {

    // Split bits with zero interleaving (21-bit support)
    inline auto split_by_3(uint32_t n) -> uint64_t {
        uint64_t x = n & 0x1fffff;  // Only use 21 bits
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
        x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }

    // Compute Morton code from 3D voxel id
    inline auto morton_code(Eigen::RowVector3i const& voxel) -> uint64_t {
        return split_by_3(voxel[0]) |
              (split_by_3(voxel[1]) << 1) |
              (split_by_3(voxel[2]) << 2);
    }

    // Return sorted indices by Morton code
    inline auto z_order_sort_indices(std::vector<Eigen::RowVector3i> const& voxels) -> std::vector<size_t> {
        std::vector<size_t> indices(voxels.size());
        std::iota(indices.begin(), indices.end(), 0);  // 0, 1, 2, ...

        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return morton_code(voxels[i]) < morton_code(voxels[j]);
        });

        return indices;
    }

    // Reorder a vector using sorted indices
    template<typename VecT>
    inline void apply_permutation_inplace(std::vector<VecT>& data, std::vector<size_t> const& indices) {
        std::vector<VecT> copy = data;
        for (size_t i = 0; i < indices.size(); ++i) {
            data[i] = copy[indices[i]];
        }
    }

}  // namespace TonSZ

#endif // Z_ORDER_CURVE_HPP
