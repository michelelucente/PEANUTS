#!/usr/bin/env python3
"""
Compares two .npz files written by workload.py. Reports, for every output, whether the values are
bit-for-bit identical, equal as numbers (which identifies +0 and -0), or different, and the largest
relative difference. Exit status 0 only if every output is equal as numbers or, with --scale-tol T,
if every difference is at most T times the largest absolute value of its output (outputs that are
exception names must match exactly).

Usage: compare.py <reference.npz> <new.npz> [--scale-tol T]
"""

import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("reference")
ap.add_argument("new")
ap.add_argument("--scale-tol", type=float, default=None)
args = ap.parse_args()
ref = np.load(args.reference)
new = np.load(args.new)

missing = sorted(set(ref.files) ^ set(new.files))
bitwise = numeric = differ = within = 0
worst = (0.0, None)
worst_scaled = (0.0, None)
for k in ref.files:
    if k not in new.files:
        continue
    a, b = ref[k], new[k]
    if a.shape != b.shape or a.dtype != b.dtype:
        differ += 1
        print(f"shape/dtype mismatch {k}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        continue
    if a.tobytes() == b.tobytes():
        bitwise += 1
    elif np.array_equal(a, b, equal_nan=True):
        numeric += 1
    else:
        differ += 1
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))) if a.dtype.kind in "fc" else float("inf")
        if rel > worst[0]:
            worst = (rel, k)
        if a.dtype.kind in "fc":
            scaled = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))
            if scaled > worst_scaled[0]:
                worst_scaled = (scaled, k)
            if args.scale_tol is not None and scaled <= args.scale_tol:
                within += 1
                continue
        if differ <= 10:
            print(f"DIFFERENT {k}: max relative difference {rel:.3e}")

print(f"outputs: {len(ref.files)}; bit-for-bit identical: {bitwise}; equal up to the sign of zero: {numeric}; "
      f"different: {differ}; missing on one side: {len(missing)}")
if differ:
    print(f"largest difference relative to the output scale {worst_scaled[0]:.3e} in {worst_scaled[1]}")
    print(f"largest difference relative to the value {worst[0]:.3e} in {worst[1]}")
if args.scale_tol is not None:
    print(f"differences within {args.scale_tol:g} of the output scale: {within}; beyond: {differ - within}")
    raise SystemExit(0 if (differ == within and not missing) else 1)
raise SystemExit(0 if (differ == 0 and not missing) else 1)
