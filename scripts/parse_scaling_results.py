#!/usr/bin/env python3
"""Aggregate per-(node, block) summary.json files into Excel.

Layout assumed:
    <root>/<base>_n<N>/blk<M>M/summary.json

For each (N, M) we read summary.json (written by XnYZip_mpi rank 0) and
derive strong-scaling metrics (speedup, parallel efficiency) per block size.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import pandas as pd


def collect(root: str, base: str) -> List[Dict]:
    node_re = re.compile(rf"^{re.escape(base)}_n(\d+)$")
    blk_re = re.compile(r"^blk(\d+)M$")
    rows: List[Dict] = []
    for entry in sorted(os.listdir(root)):
        m = node_re.match(entry)
        if not m:
            continue
        n = int(m.group(1))
        node_dir = os.path.join(root, entry)
        for blk_entry in sorted(os.listdir(node_dir)):
            bm = blk_re.match(blk_entry)
            if not bm:
                continue
            blk_mib = int(bm.group(1))
            sjson = os.path.join(node_dir, blk_entry, "summary.json")
            if not os.path.exists(sjson):
                print(f"  ! missing: {sjson}")
                continue
            with open(sjson) as f:
                s = json.load(f)
            in_b = s["input_bytes"]
            wall_max = s["wall_max_s"]
            wall_min = s["wall_min_s"]
            phase = s.get("phase_max", {})
            rows.append({
                "N_nodes": n,
                "block_size_MiB": blk_mib,
                "nranks": s.get("nranks", n),
                "num_blocks": s["num_blocks"],
                "input_GB": in_b / 1e9,
                "compressed_GB": s["compressed_bytes"] / 1e9,
                "ratio": s["ratio"],
                "wall_max_s": wall_max,
                "wall_min_s": wall_min,
                "load_balance": (wall_min / wall_max) if wall_max > 0 else None,
                "throughput_GBps": s["end_to_end_throughput_GBps"],
                "phase_read_s": phase.get("read"),
                "phase_comp_s": phase.get("comp"),
                "phase_write_s": phase.get("write"),
            })
    rows.sort(key=lambda r: (r["block_size_MiB"], r["N_nodes"]))
    return rows


def add_scaling_per_blocksize(rows: List[Dict]) -> List[Dict]:
    """Compute speedup and parallel efficiency relative to the smallest N
    available for each block size."""
    if not rows:
        return rows
    by_blk: Dict[int, List[Dict]] = {}
    for r in rows:
        by_blk.setdefault(r["block_size_MiB"], []).append(r)
    for blk, group in by_blk.items():
        group.sort(key=lambda r: r["N_nodes"])
        base = group[0]
        base_n = base["N_nodes"]
        base_thr = base["throughput_GBps"]
        for r in group:
            if base_thr and base_thr > 0:
                r["speedup"] = r["throughput_GBps"] / base_thr
                r["parallel_efficiency"] = (r["throughput_GBps"] / base_thr) / (r["N_nodes"] / base_n)
            else:
                r["speedup"] = None
                r["parallel_efficiency"] = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="run root containing <base>_nN/blkMM/")
    ap.add_argument("--base", required=True, help="base label (e.g. hacc)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")

    rows = collect(args.root, args.base)
    if not rows:
        sys.exit("no summary.json files found")
    rows = add_scaling_per_blocksize(rows)

    df = pd.DataFrame(rows)
    out = args.output or os.path.join(args.root, f"scaling_{args.base}.xlsx")
    with pd.ExcelWriter(out) as w:
        df.to_excel(w, index=False, sheet_name="raw")
        if not df.empty:
            thr = df.pivot_table(index="N_nodes", columns="block_size_MiB",
                                 values="throughput_GBps", aggfunc="first")
            ratio = df.pivot_table(index="N_nodes", columns="block_size_MiB",
                                   values="ratio", aggfunc="first")
            eff = df.pivot_table(index="N_nodes", columns="block_size_MiB",
                                 values="parallel_efficiency", aggfunc="first")
            thr.round(4).to_excel(w, sheet_name="throughput_GBps")
            ratio.round(4).to_excel(w, sheet_name="ratio")
            eff.round(4).to_excel(w, sheet_name="parallel_efficiency")
    print(f"saved: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
