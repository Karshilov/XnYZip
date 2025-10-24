#ifndef TON_SZ_BLOCK_QUANTIZER_HPP
#define TON_SZ_BLOCK_QUANTIZER_HPP

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdint>
#include <Eigen/Dense>
#include "../preprocessor/z_order_curve.hpp"
#include "./hilbert-curves-cpp/hilbert.hpp"

struct NodeWithOrder {
    uint64_t id;   
    uint64_t reid; 
    size_t   ord;   
};

struct LCPMeta {
    std::vector<uint64_t> blkst; 
    std::vector<uint64_t> blkcnt; 
    std::vector<uint8_t>  quads; 
    std::vector<uint64_t> repos; 
    std::vector<size_t> ords;
    bool is_hilbert;
};

#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE - 1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)

template <typename T>
void radix_sort(T *start, T *end) {


    size_t numElements = end - start;
    T* buffer = new T[numElements];
    int total_digits = sizeof(size_t) * 8;

    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
        size_t bucket[BASE] = {0};
        size_t offset[BASE] = {0};

        for(size_t i = 0; i < numElements; i++){
            bucket[DIGITS(start[i].id, shift)]++;
        }

        for (size_t i = 1; i < BASE; i++) {
            offset[i] = offset[i - 1] + bucket[i - 1];
        }

        for(size_t i = 0; i < numElements; i++) {
            size_t cur_num_digit = DIGITS(start[i].id, shift);
            size_t pos = offset[cur_num_digit]++;
            buffer[pos] = start[i];
        }

        T* tmp = start;
        start = buffer;
        buffer = tmp;
    }

    delete[] buffer;

    T *l = start, *r = l;
    while(l < end){
        r = l;
        while(r + 1 < end && l -> id == (r + 1) -> id){
            ++r;
        }
        if(l < r) std::sort(l, r + 1, [&](T u, T v){return u.reid < v.reid;});
        l = r + 1;
    }

}

/**
 * @param pts  
 * @param bx,by,bz 
 * @param nx,ny,nz 
 * @returns 
 */
template <typename T>
auto block_quantize(
    std::vector<Eigen::RowVector3<T> >& pts,
    size_t bx, size_t by, size_t bz,
    size_t nx, size_t ny, size_t nz,
    bool is_hilbert) -> LCPMeta
{
    size_t n = pts.size();
    std::vector<NodeWithOrder> vec(n);
    nx = nx / (2 * bx) + 1;
    ny = ny / (2 * by) + 1;
    nz = nz / (2 * bz) + 1;

    for (size_t i = 0; i < n; ++i) {
        auto &p = pts[i];
        size_t cx = p.x() / bx;
        size_t dx = p.x() % bx;
        size_t cy = p.y() / by;
        size_t dy = p.y() % by;
        size_t cz = p.z() / bz;
        size_t dz = p.z() % bz;
        uint64_t id   = (cx / 2) + (cy / 2) * nx + (cz / 2) * nx * ny;
        // vec[i] = { .id=id, .reid=(dx + dy * bx + dz * bx * by) | ((cx & 1) << 60) | ((cy & 1) << 61) |
        //                               ((cz & 1) << 62), .ord=i };
        if (!is_hilbert) {
            vec[i] = { .id=id, .reid=XnYZip::morton_code(Eigen::RowVector3i{static_cast<int>(dx), static_cast<int>(dy), static_cast<int>(dz)}) | ((cx & 1) << 60) | ((cy & 1) << 61) |
                                      ((cz & 1) << 62), .ord=i };
        } else {
            vec[i] = { .id=id, .reid=mve::mortonToHilbert3(XnYZip::morton_code(Eigen::RowVector3i{static_cast<int>(dx), static_cast<int>(dy), static_cast<int>(dz)}), 20) | ((cx & 1) << 60) | ((cy & 1) << 61) |
                                      ((cz & 1) << 62), .ord=i };
        }
    }
    radix_sort<NodeWithOrder>(vec.data(), vec.data() + n);
    size_t blknum = 1;
    for (size_t i = 1; i < n; ++i)
        if (vec[i].id != vec[i-1].id)
            ++blknum;
    LCPMeta M;
    M.is_hilbert = is_hilbert;
    M.blkst .resize(blknum);
    M.blkcnt.resize(blknum);
    M.quads .resize(n);
    M.repos .resize(n);
    M.ords  .resize(n);
    size_t i = -1;
    uint64_t prevId   = uint64_t(-1);
    uint64_t prevQuad = 0, prevRe = 0;
    for (size_t j = 0; j < n; ++j) {
        auto &node = vec[j];
        M.ords[j] = node.ord;
        uint64_t id   = node.id;
        size_t quad = node.reid >> 60;
        size_t reid = node.reid & 0x0fffffffffffffff;
        if (id != prevId) {
            M.blkst[++i] = prevId = id;
            prevQuad = 0;
            prevRe = 0;
        } else if (quad != prevQuad) {
            prevRe = 0;
        }
        M.blkcnt[i] += 1;
        M.quads[j] = uint8_t(quad - prevQuad);
        M.repos[j] =  reid   - prevRe;
        prevQuad = quad;
        prevRe   = reid;
    }
    return M;
}

template <typename T>
auto recover_from_lcp_meta(
    const LCPMeta& meta,
    size_t bx, size_t by, size_t bz,
    size_t nx, size_t ny, size_t nz
) -> std::vector<Eigen::RowVector3<T>>
{
    std::vector<Eigen::RowVector3<T>> coords;
    coords.reserve(meta.repos.size());

    nx = nx / (2 * bx) + 1;
    ny = ny / (2 * by) + 1;
    nz = nz / (2 * bz) + 1;

    auto [blkst, blkcnt, quads, repos, ords, is_hilbert] = meta;

    int i = 0, j = 0;
    for (; i < blkst.size(); i++) {

                size_t pbx = (blkst[i] % nx * 2) * bx;
                size_t pby = (blkst[i] / nx % ny * 2) * by;
                size_t pbz = (blkst[i] / nx / ny * 2) * bz;

                size_t prequad = 0;
                size_t prerepos = 0;


                 for (size_t j_ = 0; j_ < blkcnt[i]; j_++) {

                    if (quads[j] != 0) prerepos = 0;
                    size_t reposj = repos[j] + prerepos;
                    size_t quadj = quads[j] + prequad;
                    prerepos = reposj;
                    prequad = quadj;

                    // Eigen::RowVector3i decoded = XnYZip::decode_morton_code(reposj);
                    // uint64_t decoded_x, decoded_y, decoded_z;
                    // hilbert_decode(reposj, decoded_x, decoded_y, decoded_z);
                    auto morton = is_hilbert ? mve::hilbertToMorton3(reposj, 20) : reposj;
                    auto decoded = XnYZip::decode_morton_code(morton);

                    size_t idx = (pbx + ((quadj & 0x01) >> 0) * bx + static_cast<size_t>(decoded[0]));
                    size_t idy = (pby + ((quadj & 0x02) >> 1) * by + static_cast<size_t>(decoded[1]));
                    size_t idz = (pbz + ((quadj & 0x04) >> 2) * bz + static_cast<size_t>(decoded[2]));
                    // size_t idx = (pbx + ((quadj & 0x01) >> 0) * bx + (reposj % bx));
                    // size_t idy = (pby + ((quadj & 0x02) >> 1) * by + (reposj / bx % by));
                    // size_t idz = (pbz + ((quadj & 0x04) >> 2) * bz + (reposj / bx / by));

                    coords.emplace_back(Eigen::RowVector3<T>{static_cast<T>(idx), static_cast<T>(idy), static_cast<T>(idz)});

                    ++j;
                }
            }
    return coords;
}


#endif // TON_SZ_BLOCK_QUANTIZER_HPP