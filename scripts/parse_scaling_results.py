#!/usr/bin/env python3
"""Aggregate per-node summary.json files from a scaling sweep into Excel.

Layout assumed:
    <root>/n1/summary.json
    <root>/n2/summary.json
    <root>/n4/summary.json
    ...

Each summary.json is what XnYZip_mpi writes on rank 0. We collect them, derive
strong-scaling metrics (speedup, parallel_efficiency w.r.t. the smallest N
present), and write an Excel with raw + summary sheets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import pandas as pd

NODE_DIR_RE = re.compile(r"^n(\d+)$")


def collect(root: str) -> List[Dict]:
    rows: List[Dict] = []
    for entry in sorted(os.listdir(root)):
        m = NODE_DIR_RE.match(entry)
        if not m:
            continue
        n = int(m.group(1))
        path = os.path.join(root, entry, "summary.json")
        if not os.path.exists(path):
            print(f"  ! missing: {path}")
            continue
        with open(path) as f:
            s = json.load(f)
        ranks = s.get("nranks", n)
        in_b = s["input_bytes"]
        out_b = s["compressed_bytes"]
        wall_max = s["wall_max_s"]
        wall_min = s["wall_min_s"]
        phase = s.get("phase_max", {})
        rows.append({
            "N_nodes": n,
            "nranks": ranks,
            "num_blocks": s["num_blocks"],
            "block_size_MiB": s["block_size"] / (1024 * 1024),
            "input_GB": in_b / 1e9,
            "compressed_GB": out_b / 1e9,
            "ratio": s["ratio"],
            "wall_max_s": wall_max,
            "wall_min_s": wall_min,
            "load_balance": (wall_min / wall_max) if wall_max > 0 else None,
            "throughput_GBps": s["end_to_end_throughput_GBps"],
            "phase_read_s": phase.get("read"),
            "phase_comp_s": phase.get("comp"),
            "phase_write_s": phase.get("write"),
        })
    rows.sort(key=lambda r: r["N_nodes"])
    return rows


def add_scaling_columns(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return rows
    base = rows[0]
    base_n = base["N_nodes"]
    base_thr = base["throughput_GBps"]
    for r in rows:
        if base_thr and base_thr > 0:
            r["speedup_vs_n"+str(base_n)] = r["throughput_GBps"] / base_thr
            r["parallel_efficiency"] = r["throughput_GBps"] / base_thr / (r["N_nodes"] / base_n)
        else:
            r["speedup_vs_n"+str(base_n)] = None
            r["parallel_efficiency"] = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing n1/, n2/, n4/, ...")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")

    rows = collect(args.root)
    if not rows:
        sys.exit("no summary.json files found")
    rows = add_scaling_columns(rows)

    df = pd.DataFrame(rows)
    out = args.output or os.path.join(args.root, "scaling.xlsx")
    with pd.ExcelWriter(out) as w:
        df.to_excel(w, index=False, sheet_name="raw")
        if "ratio" in df.columns:
            df[["N_nodes", "ratio"]].to_excel(w, index=False, sheet_name="ratio_by_N")
        if "throughput_GBps" in df.columns:
            df[["N_nodes", "throughput_GBps", "parallel_efficiency"]].to_excel(
                w, index=False, sheet_name="scaling"
            )
    print(f"saved: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
