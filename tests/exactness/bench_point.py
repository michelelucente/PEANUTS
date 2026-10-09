#!/usr/bin/env python3
"""
Cost of one likelihood point with the PEANUTS call pattern of the GAMBIT likelihoods, excluding the
JIT compilation (a first point is evaluated and discarded). Every call goes through the same Python
entry points as the frontend (Backends/src/frontends/PEANUTS_1_5.cpp): a fresh PMNS object per
Pearth_integrated call, the solar model columns fetched per solar_weights call.

  sno       : 201 energies in [5, 15] MeV, solar_weights(8B) + Pearth_integrated(day) and (night)
              at each, depth 1730 m, SnoCosZenith.dat (NeutrinoBit Oscillations_SNO.cpp)
  borexino  : 256 master energies in [0.05, 22] MeV, Pearth_integrated for the three mass states
              (all nadir angles), depth 1400 m, LNGSCosZenith.dat; plus 200 solar_weights nodes for
              each of the six continuous sources and the three line energies (Oscillations_Borexino.cpp)

Usage: bench_point.py <peanuts root> <sno|borexino> [--scale F] [--exposure-dir DIR] [--points N] [--out file.npz]
--scale F multiplies the number of energies and nodes (F < 1 for a quick local estimate).
"""

import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("case", choices=["sno", "borexino"])
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--exposure-dir", default=None)
ap.add_argument("--points", type=int, default=2)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.root))
import peanuts
from peanuts.solar import solar_flux_mass

sm = peanuts.SolarModel()
ed = peanuts.EarthDensity()

def solar_weights(th12, th13, dm21, dm3l, E, frac):
    return np.asarray(solar_flux_mass(th12, th13, dm21, dm3l, E, sm.radius(), sm.density(), sm.fraction(frac)), dtype=np.float64)

def pearth_integrated(state, th12, th13, th23, d, dm21, dm3l, E, H, file, daynight):
    pmns = peanuts.PMNS(th12, th13, th23, d)
    return peanuts.Pearth_integrated(state, ed, pmns, dm21, dm3l, E, H, ns=480, normalized=False,
                                     from_file=file, angle="CosZenith", daynight=daynight)

def sno_point(p, out):
    th12, th13, th23, d, dm21, dm3l = p
    file = os.path.join(args.root, "Data", "SnoCosZenith.dat")
    n = max(3, int(round(201 * args.scale)))
    for E in np.linspace(5.0, 15.0, n):
        for dn in ["day", "night"]:
            w = solar_weights(th12, th13, dm21, dm3l, E, "8B")
            out.append(pearth_integrated(w, th12, th13, th23, d, dm21, dm3l, E, 1730.0, file, dn))

def borexino_point(p, out):
    th12, th13, th23, d, dm21, dm3l = p
    file = os.path.join(args.exposure_dir, "LNGSCosZenith.dat")
    n = max(3, int(round(256 * args.scale)))
    lines = [0.3843, 0.8613, 1.44]
    for E in list(np.geomspace(0.05, 22.0, n)) + lines:
        for j in range(3):
            s = np.zeros(3); s[j] = 1.0
            out.append(pearth_integrated(s, th12, th13, th23, d, dm21, dm3l, E, 1400.0, file, ""))
    nodes = max(2, int(round(200 * args.scale)))
    ranges = {"pp": (0.01, 0.42), "13N": (0.01, 1.2), "15O": (0.01, 1.73), "17F": (0.01, 1.74), "8B": (0.02, 16.5), "hep": (0.02, 18.7)}
    for src, (lo, hi) in ranges.items():
        for E in np.linspace(lo, hi, nodes):
            out.append(solar_weights(th12, th13, dm21, dm3l, E, src))
    for src, E in [("7Be", 0.3843), ("7Be", 0.8613), ("pep", 1.44)]:
        out.append(solar_weights(th12, th13, dm21, dm3l, E, src))

pts = [(0.5903, 0.1503, 0.8587, 3.4034, 7.42e-5, 2.51e-3), (0.5850, 0.1490, 0.8500, 3.3, 7.60e-5, 2.50e-3),
       (0.6000, 0.1480, 0.8400, 3.2, 7.30e-5, -2.48e-3), (0.5700, 0.1510, 0.8700, 3.5, 7.50e-5, 2.52e-3)]
run = sno_point if args.case == "sno" else borexino_point
times, outputs = [], []
for i in range(args.points):
    out = []
    t0 = time.perf_counter()
    run(pts[i % len(pts)], out)
    times.append(time.perf_counter() - t0)
    outputs.append(np.array(out))
print(f"{args.case} scale={args.scale}: first point (with JIT) {times[0]:.2f} s; "
      f"subsequent points {', '.join(f'{t:.3f}' for t in times[1:])} s")
if args.out:
    np.savez(args.out, *outputs, times=np.array(times))
