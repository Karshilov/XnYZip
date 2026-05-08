#!/usr/bin/env python3
"""Pre-sort a point-cloud .bin (xyz interleaved float32) by Morton code.

Used as a preprocessing step before distributed XnYZip compression on
non-raster inputs (LIDAR / particle / sampled point clouds where file
order does NOT correspond to a 3D-space partition). After sorting, each
chunk taken at file-byte granularity becomes a spatially-disjoint
sub-region, which eliminates the cross-chunk dedup-table redundancy
that would otherwise waste 40-50% of compressed size.

NOT NEEDED for genuinely raster-ordered inputs (file-adjacent =
spatially-adjacent). For those, file-order chunking already gives
≥97% of monolithic ratio.

The default sort key uses the float32 bit pattern directly, making the
preprocessing **bound-independent** — the same sorted file works for
any compression error bound.

Usage:
    python scripts/morton_sort.py --input data.bin --output data_morton.bin

Optional:
    --key-mode {floatbit, quantize}   Default: floatbit
    --bound-scale 1e-3                Only used in quantize mode
"""

import argparse
import os
import sys
import time

import numpy as np


def spread_bits_3d(v: np.ndarray, n_bits: int) -> np.ndarray:
    """Spread the low n_bits of each value with two zero bits between bits.

    For n_bits up to 21 this fits inside uint64 (using 63 of 64 bits).
    """
    if n_bits > 21:
        raise ValueError("Morton code with > 21 bits/dim does not fit uint64")
    v = v.astype(np.uint64) & ((1 << n_bits) - 1)
    v = (v | (v << 32)) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << 16)) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << 8))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << 4))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << 2))  & np.uint64(0x1249249249249249)
    return v


def morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_bits: int) -> np.ndarray:
    return (spread_bits_3d(x, n_bits)
            | (spread_bits_3d(y, n_bits) << np.uint64(1))
            | (spread_bits_3d(z, n_bits) << np.uint64(2)))


def morton_sort_file(in_path: str, out_path: str,
                     key_mode: str = 'floatbit',
                     bound_scale: float = 1e-3,
                     n_bits: int = 21) -> None:
    t0 = time.time()
    pts = np.fromfile(in_path, dtype=np.float32).reshape(-1, 3)
    print(f"  loaded {len(pts):,} points from {in_path} in {time.time()-t0:.2f}s")

    if key_mode == 'floatbit':
        # Bound-INDEPENDENT: sort key derived from float32 bit pattern.
        # For non-negative floats, integer-comparison on bit pattern equals
        # numeric comparison. We right-shift to fit each axis in n_bits, which
        # discards low-precision fraction bits but preserves the ordering up
        # to 2^(23-shift) significant bits. This is more than enough to
        # determine 3D locality for chunking purposes.
        if (pts < 0).any():
            raise ValueError("floatbit mode requires all-non-negative coords; "
                             "use --key-mode quantize for signed data.")
        fb = pts.view(np.uint32)
        max_v = int(fb.max())
        if max_v == 0:
            shift = 0
        else:
            shift = max(0, max_v.bit_length() - n_bits)
        print(f"  floatbit max={max_v}, shift={shift} → effective n_bits per dim = {n_bits}")
        v = (fb >> shift).astype(np.uint64)
        qx, qy, qz = v[:, 0], v[:, 1], v[:, 2]

    elif key_mode == 'quantize':
        # Bound-aware: same cell size as XnYZip's cube quantizer at the given
        # bound. Sort key is suboptimal if compressing at a much tighter eb
        # later, but in practice the resulting order is still good enough.
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        bound = bound_scale * diag
        cell = 2.0 * bound / np.sqrt(3.0)
        print(f"  quantize: diag={diag:.4f}, bound={bound:.6f}, cell={cell:.6f}")
        qpts = np.floor(pts / cell).astype(np.int64)
        qx = (qpts[:, 0] - qpts[:, 0].min()).astype(np.uint64)
        qy = (qpts[:, 1] - qpts[:, 1].min()).astype(np.uint64)
        qz = (qpts[:, 2] - qpts[:, 2].min()).astype(np.uint64)
        max_range = int(max(qx.max(), qy.max(), qz.max())) + 1
        if max_range >= (1 << n_bits):
            raise ValueError(
                f"quantized range {max_range} exceeds {1 << n_bits}. Use floatbit mode.")
    else:
        raise ValueError(f"unknown key-mode: {key_mode}")

    t2 = time.time()
    codes = morton3(qx, qy, qz, n_bits)
    print(f"  morton encode: {time.time()-t2:.2f}s")

    t3 = time.time()
    order = np.argsort(codes, kind='stable')
    print(f"  sort: {time.time()-t3:.2f}s")

    t4 = time.time()
    pts[order].tofile(out_path)
    print(f"  write {os.path.getsize(out_path):,} bytes to {out_path} in {time.time()-t4:.2f}s")
    print(f"  total elapsed: {time.time()-t0:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--key-mode', choices=['floatbit', 'quantize'], default='floatbit',
                    help='floatbit: bound-independent sort using float32 bit pattern '
                         '(default, recommended). quantize: bound-aware sort using '
                         '--bound-scale; gives slightly better ordering at one specific bound.')
    ap.add_argument('--bound-scale', type=float, default=1e-3,
                    help='only used with --key-mode quantize (default 1e-3)')
    ap.add_argument('--bits', type=int, default=21,
                    help='bits per dimension for Morton code (max 21 → fits uint64)')
    args = ap.parse_args()
    morton_sort_file(args.input, args.output, args.key_mode, args.bound_scale, args.bits)


if __name__ == '__main__':
    main()
