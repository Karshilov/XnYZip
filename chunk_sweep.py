#!/usr/bin/env python3
"""Chunk-size sensitivity sweep for XnYZip.

Splits the input bin file into fixed-size byte chunks via the new
``--offset=BYTES --count=BYTES`` flags, runs the binary on each chunk,
sums compressed bytes, and reports the global compression ratio per
(chunk_size, variant). Also runs the full-file ("monolithic") baseline
so the paper can quote ratio loss as a function of chunk size.

Usage:
    python chunk_sweep.py --input /path/to/data.bin
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

POINT_BYTES = 12  # float32 * 3


def run_cmd(cmd: str, timeout: Optional[int] = None) -> Tuple[str, str]:
    """Run a shell command, capture stdout/stderr to temp files (WSL-safe)."""
    print(f"🚀 {cmd}")
    fd_out, path_out = tempfile.mkstemp(suffix=".stdout", text=True)
    fd_err, path_err = tempfile.mkstemp(suffix=".stderr", text=True)
    try:
        with os.fdopen(fd_out, "w") as f_out, os.fdopen(fd_err, "w") as f_err:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=f_out,
                stderr=f_err,
                text=True,
                timeout=timeout,
            )
        with open(path_out) as f:
            stdout = f.read()
        with open(path_err) as f:
            stderr = f.read()
        return stdout, stderr
    except subprocess.CalledProcessError:
        with open(path_out) as f:
            stdout = f.read()
        with open(path_err) as f:
            stderr = f.read()
        print(f"❌ command failed\n   stdout tail: {stdout[-400:]}\n   stderr tail: {stderr[-400:]}")
        raise
    finally:
        try:
            os.unlink(path_out)
            os.unlink(path_err)
        except OSError:
            pass


@dataclass(frozen=True)
class Variant:
    name: str
    quantizer: str   # "cube" | "to"
    curve: str       # "-z" | "-h"
    rle: str         # "-rle" | "-normal"


DEFAULT_VARIANTS: List[Variant] = [
    Variant("to-z-normal", "to", "-z", "-normal"),
    Variant("to-h-normal", "to", "-h", "-normal"),
    Variant("to-z-rle",    "to", "-z", "-rle"),
    Variant("to-h-rle",    "to", "-h", "-rle"),
]


def parse_compressed_bytes(stdout: str) -> Optional[int]:
    m = re.search(r"chunk\s*compressed\s*bytes\s*:\s*(\d+)", stdout, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_input_bytes(stdout: str) -> Optional[int]:
    m = re.search(r"chunk\s*input\s*bytes\s*:\s*(\d+)", stdout, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_psnr(stdout: str) -> Optional[float]:
    m = re.search(r"p2p\s*psnr\s*:\s*([-\d.]+)", stdout, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_compression_ms(stdout: str) -> Optional[float]:
    m = re.search(r"XnYZip\s*compression[^0-9]*([\d.]+)\s*ms", stdout, re.IGNORECASE)
    return float(m.group(1)) if m else None


def chunk_byte_offsets(file_size: int, chunk_bytes: int) -> List[Tuple[int, int]]:
    """Return [(offset, count), ...] tiling the file. Last chunk may be smaller.

    `count` is always a multiple of POINT_BYTES (truncates trailing partial point).
    """
    chunk_bytes = chunk_bytes - (chunk_bytes % POINT_BYTES)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be >= POINT_BYTES")
    out = []
    aligned_total = file_size - (file_size % POINT_BYTES)
    off = 0
    while off < aligned_total:
        cnt = min(chunk_bytes, aligned_total - off)
        out.append((off, cnt))
        off += cnt
    return out


def diagonal_of(bin_path: str) -> float:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def run_one(
    exe: str,
    bin_path: str,
    bound: float,
    variant: Variant,
    offset: Optional[int],
    count: Optional[int],
    decomp_path: str,
    direct_threshold: int,
) -> Dict:
    """Invoke XnYZip once. offset/count=None means full-file mode."""
    cmd = (
        f"{exe} {bin_path} {variant.quantizer} {bound} {variant.curve} "
        f"{variant.rle} {decomp_path} {direct_threshold}"
    )
    if offset is not None:
        cmd += f" --offset={offset}"
    if count is not None:
        cmd += f" --count={count}"

    t0 = time.perf_counter()
    stdout, _ = run_cmd(cmd)
    wall = time.perf_counter() - t0

    return {
        "input_bytes": parse_input_bytes(stdout),
        "compressed_bytes": parse_compressed_bytes(stdout),
        "psnr": parse_psnr(stdout),
        "compress_ms": parse_compression_ms(stdout),
        "wall_s": wall,
    }


def sweep(
    exe: str,
    bin_path: str,
    bound: float,
    chunk_sizes_mib: List[float],
    variants: List[Variant],
    direct_threshold: int,
    out_xlsx: str,
    temp_dir: str,
) -> pd.DataFrame:
    file_size = os.path.getsize(bin_path)
    file_size_aligned = file_size - (file_size % POINT_BYTES)
    print(f"input file {bin_path}: {file_size} bytes, {file_size_aligned // POINT_BYTES} points (aligned)")

    rows: List[Dict] = []

    # ---- 1. Monolithic baseline (chunk_size == "full") ----
    for v in variants:
        decomp = os.path.join(temp_dir, f"full_{v.name}.f32")
        print(f"\n=== Monolithic baseline: variant={v.name} ===")
        try:
            r = run_one(exe, bin_path, bound, v, None, None, decomp, direct_threshold)
        except subprocess.CalledProcessError:
            r = {"input_bytes": None, "compressed_bytes": None, "psnr": None, "compress_ms": None, "wall_s": None}
        rows.append({
            "chunk_size_mib": "full",
            "chunk_bytes": file_size_aligned,
            "num_chunks": 1,
            "variant": v.name,
            "total_input_bytes": r["input_bytes"] or file_size_aligned,
            "total_compressed_bytes": r["compressed_bytes"],
            "ratio": (
                r["input_bytes"] / r["compressed_bytes"]
                if (r["input_bytes"] and r["compressed_bytes"]) else None
            ),
            "mean_per_chunk_psnr": r["psnr"],
            "total_compress_ms": r["compress_ms"],
            "total_wall_s": r["wall_s"],
        })
        gc.collect()

    # ---- 2. Chunked sweep ----
    for size_mib in chunk_sizes_mib:
        chunk_bytes = int(size_mib * 1024 * 1024)
        chunks = chunk_byte_offsets(file_size_aligned, chunk_bytes)
        print(f"\n=== chunk_size={size_mib} MiB → {len(chunks)} chunk(s) ===")
        for v in variants:
            print(f"--- variant={v.name} ---")
            tot_in = 0
            tot_out = 0
            psnrs: List[float] = []
            tot_compress_ms = 0.0
            tot_wall = 0.0
            failed = False
            for idx, (off, cnt) in enumerate(chunks):
                decomp = os.path.join(temp_dir, f"chunk_{v.name}_{size_mib}_{idx}.f32")
                try:
                    r = run_one(exe, bin_path, bound, v, off, cnt, decomp, direct_threshold)
                except subprocess.CalledProcessError:
                    failed = True
                    break
                if r["input_bytes"] is None or r["compressed_bytes"] is None:
                    failed = True
                    break
                tot_in += r["input_bytes"]
                tot_out += r["compressed_bytes"]
                if r["psnr"] is not None and not math.isinf(r["psnr"]):
                    psnrs.append(r["psnr"])
                if r["compress_ms"] is not None:
                    tot_compress_ms += r["compress_ms"]
                tot_wall += r["wall_s"] or 0.0
                # cleanup decomp file to keep tmp small
                try: os.unlink(decomp)
                except OSError: pass
            rows.append({
                "chunk_size_mib": size_mib,
                "chunk_bytes": chunk_bytes,
                "num_chunks": len(chunks),
                "variant": v.name,
                "total_input_bytes": tot_in if not failed else None,
                "total_compressed_bytes": tot_out if not failed else None,
                "ratio": (tot_in / tot_out) if (not failed and tot_out > 0) else None,
                "mean_per_chunk_psnr": (float(np.mean(psnrs)) if psnrs else None),
                "total_compress_ms": tot_compress_ms,
                "total_wall_s": tot_wall,
            })
            # save checkpoint after each (size, variant)
            df = pd.DataFrame(rows)
            with pd.ExcelWriter(out_xlsx) as w:
                df.to_excel(w, index=False, sheet_name="raw")
            gc.collect()

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="chunk_size_mib", columns="variant", values="ratio", aggfunc="first"
    )
    with pd.ExcelWriter(out_xlsx) as w:
        df.to_excel(w, index=False, sheet_name="raw")
        pivot.round(4).to_excel(w, sheet_name="ratio_pivot")
    print(f"\n✅ saved: {out_xlsx}")
    print(pivot.round(4))
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input .bin file (float32, xyz interleaved)")
    p.add_argument("--exe", default="./build/XnYZip")
    p.add_argument(
        "--bound-scale", type=float, default=1e-3,
        help="L2 bound = bound_scale * bbox diagonal of the FULL file (kept constant across chunks)",
    )
    p.add_argument("--bound", type=float, default=None, help="absolute L2 bound (overrides --bound-scale)")
    p.add_argument(
        "--chunk-sizes-mib", type=float, nargs="+",
        default=[1, 4, 16, 64, 256],
        help="chunk sizes in MiB to sweep",
    )
    p.add_argument("--direct-threshold", type=int, default=1024)
    p.add_argument("--output", default=None, help="output xlsx path")
    p.add_argument("--temp-dir", default=None, help="temp dir for decomp files")
    p.add_argument(
        "--variants", nargs="+", default=None,
        help=f"variant names (subset of {[v.name for v in DEFAULT_VARIANTS]})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    if not os.path.exists(args.exe):
        sys.exit(f"binary not found: {args.exe}")

    base = os.path.splitext(os.path.basename(args.input))[0]
    out = args.output or f"chunk_sweep_{base}.xlsx"
    temp_dir = args.temp_dir or f"{base}_chunk_sweep_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    if args.bound is not None:
        bound = args.bound
    else:
        diag = diagonal_of(args.input)
        bound = args.bound_scale * diag
        print(f"diagonal={diag:.6f}, bound_scale={args.bound_scale} → bound={bound:.6f}")

    variants = DEFAULT_VARIANTS
    if args.variants:
        chosen = set(args.variants)
        variants = [v for v in DEFAULT_VARIANTS if v.name in chosen]
        if not variants:
            sys.exit(f"no matching variants in {args.variants}")

    sweep(
        exe=args.exe,
        bin_path=args.input,
        bound=bound,
        chunk_sizes_mib=args.chunk_sizes_mib,
        variants=variants,
        direct_threshold=args.direct_threshold,
        out_xlsx=out,
        temp_dir=temp_dir,
    )


if __name__ == "__main__":
    main()
