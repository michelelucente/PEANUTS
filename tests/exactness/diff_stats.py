#!/usr/bin/env python3
"""
Size of the differences between two workload.py outputs: for every output that differs, the
largest absolute difference, the largest difference relative to the largest absolute value of
that output (the scale of the probabilities it contains), and the values involved.

Usage: diff_stats.py <reference.npz> <new.npz>
"""

import sys
import numpy as np

ref, new = np.load(sys.argv[1]), np.load(sys.argv[2])
rows = []
for k in ref.files:
    a, b = ref[k], new[k]
    if a.dtype.kind not in "fc" or a.tobytes() == b.tobytes():
        continue
    d = np.abs(a - b)
    i = int(np.argmax(d))
    scale = float(np.max(np.abs(a)))
    rows.append((k, float(d.flat[i]), float(d.flat[i]) / scale, float(abs(a.flat[i])), scale, a.flat[i], b.flat[i]))

numeric = [k for k in ref.files if ref[k].dtype.kind in "fc"]
print(f"numeric outputs: {len(numeric)}; differing: {len(rows)}")
if rows:
    absd = np.array([r[1] for r in rows]); scaled = np.array([r[2] for r in rows])
    print(f"max |difference|                          : {absd.max():.3e}")
    print(f"max |difference| / max|output|            : {scaled.max():.3e}")
    print(f"median |difference| / max|output|         : {np.median(scaled):.3e}")
    print(f"differences / max|output| above 1e-15     : {(scaled > 1e-15).sum()}, above 1e-13: {(scaled > 1e-13).sum()}")
    print("\nlargest five by |difference| / max|output|:")
    for r in sorted(rows, key=lambda r: -r[2])[:5]:
        print(f"  {r[0]}: |diff| {r[1]:.3e}, at |value| {r[3]:.3e}, output scale {r[4]:.3e}; ref {r[5]!r}, new {r[6]!r}")
    print("\nlargest five relative to the value itself:")
    for r in sorted(rows, key=lambda r: -(r[1] / max(r[3], 1e-300)))[:5]:
        print(f"  {r[0]}: |diff| {r[1]:.3e} on |value| {r[3]:.3e} (relative {r[1]/max(r[3],1e-300):.3e}); output scale {r[4]:.3e}")
    cats = {}
    for r in rows:
        c = r[0].split("/")[0] + ("/" + r[0].split("/")[-1] if r[0].startswith("Pearth_integrated/") else "")
        cats[c] = cats.get(c, 0) + 1
    print("\ndiffering outputs by kind:", dict(sorted(cats.items())))
