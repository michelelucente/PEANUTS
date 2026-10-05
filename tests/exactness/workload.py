#!/usr/bin/env python3
"""
Exactness workload: evaluates the PEANUTS entry points with the call pattern of the GAMBIT frontend
(Backends/src/frontends/PEANUTS_1_5.cpp) over a deterministic set of parameter points, and stores
every output in an .npz file. Two runs, one with a reference PEANUTS and one with a modified one,
are compared bit by bit with compare.py.

Usage:
  workload.py <peanuts root> <output.npz> [--size small|full] [--exposure-dir DIR]

<peanuts root> is the directory that contains the `peanuts` package. The exposure files used are
SnoCosZenith.dat (shipped with PEANUTS) and, if found in --exposure-dir, LNGSCosZenith.dat (the
Borexino exposure of NeutrinoBit).
"""

import argparse
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("output")
ap.add_argument("--size", default="small", choices=["small", "full"])
ap.add_argument("--exposure-dir", default=None)
ap.add_argument("--seed", type=int, default=20261005)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.root))
import peanuts
from peanuts.earth import Pearth
from peanuts.solar import solar_flux_mass, Psolar

assert os.path.dirname(peanuts.__file__) == os.path.join(os.path.abspath(args.root), "peanuts"), peanuts.__file__

rng = np.random.default_rng(args.seed)
results = {}
timing = {}

def store(name, value):
    results[name] = np.asarray(value)

def store_call(name, f, *a, **kw):
    """Stores the result of f(*a, **kw), or the name of the exception it raises."""
    try:
        store(name, f(*a, **kw))
    except Exception as e:
        store(name, np.array("exception:" + type(e).__name__))

# Parameter points: (th12, th13, th23, delta, dm21, dm3l). Normal and inverted ordering, the LMA
# region and points far from it, including theta13 = 0 and delta = 0.
def points(n):
    pts = [
        (0.5903, 0.1503, 0.8587, 3.4034, 7.42e-5, 2.51e-3),
        (0.5903, 0.1503, 0.8587, 3.4034, 7.42e-5, -2.49e-3),
        (0.5800, 0.0, 0.7854, 0.0, 5.0e-5, 2.5e-3),
        (0.6100, 0.1500, 0.9000, 4.7, 1.5e-4, -2.4e-3),
    ]
    for _ in range(n - len(pts)):
        dm3l = rng.uniform(2.2e-3, 2.7e-3) * rng.choice([-1.0, 1.0])
        pts.append((rng.uniform(0.45, 0.75), rng.uniform(0.0, 0.25), rng.uniform(0.6, 1.0),
                    rng.uniform(0, 2*np.pi), 10**rng.uniform(-5.5, -3.5), dm3l))
    return pts

full = args.size == "full"
pts = points(12 if full else 4)

t0 = time.time()
solar_model = peanuts.SolarModel()
earth_density = peanuts.EarthDensity()
timing["setup"] = time.time() - t0

# 1. solar_weights and Psolar (frontend: solar_flux_mass with radius, density, fraction)
t0 = time.time()
radius, density = solar_model.radius(), solar_model.density()
fractions = ["pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F"]
energies_solar = np.geomspace(0.05, 20.0, 25 if full else 7)
for ip, (th12, th13, th23, d, dm21, dm3l) in enumerate(pts):
    for fr in fractions:
        frac = solar_model.fraction(fr)
        for iE, E in enumerate(energies_solar):
            store(f"solar_weights/{ip}/{fr}/{iE}", solar_flux_mass(th12, th13, dm21, dm3l, E, radius, density, frac))
    pmns = peanuts.PMNS(th12, th13, th23, d)
    for iE, E in enumerate(energies_solar):
        store(f"Psolar/{ip}/{iE}", Psolar(pmns, dm21, dm3l, E, radius, density, solar_model.fraction("8B")))
timing["solar"] = time.time() - t0

# 2. Pearth_integrated with the frontend's arguments: ns=480, CosZenith exposure files
exposures = [("SNO", os.path.join(args.root, "Data", "SnoCosZenith.dat"), 1730.0)]
if args.exposure_dir and os.path.isfile(os.path.join(args.exposure_dir, "LNGSCosZenith.dat")):
    exposures.append(("LNGS", os.path.join(args.exposure_dir, "LNGSCosZenith.dat"), 1400.0))

energies_earth = np.geomspace(0.05, 22.0, 12 if full else 4)
t0 = time.time()
ncalls = 0
for ip, (th12, th13, th23, d, dm21, dm3l) in enumerate(pts):
    for name, file, H in exposures:
        for iE, E in enumerate(energies_earth):
            frac = solar_model.fraction("8B")
            w = np.asarray(solar_flux_mass(th12, th13, dm21, dm3l, E, radius, density, frac), dtype=np.float64)
            for daynight in ["day", "night", ""]:
                # SNO pattern: production-averaged mass weights
                pmns = peanuts.PMNS(th12, th13, th23, d)
                store(f"Pearth_integrated/{ip}/{name}/{iE}/{daynight}/w",
                      peanuts.Pearth_integrated(w, earth_density, pmns, dm21, dm3l, E, H, ns=480, normalized=False,
                                                from_file=file, angle="CosZenith", daynight=daynight))
                ncalls += 1
                # Borexino pattern: the three mass states in a row, a fresh PMNS object each time
                for j in range(3):
                    state = np.zeros(3)
                    state[j] = 1.0
                    pmns = peanuts.PMNS(th12, th13, th23, d)
                    store(f"Pearth_integrated/{ip}/{name}/{iE}/{daynight}/e{j}",
                          peanuts.Pearth_integrated(state, earth_density, pmns, dm21, dm3l, E, H, ns=480, normalized=False,
                                                    from_file=file, angle="CosZenith", daynight=daynight))
                    ncalls += 1
timing["Pearth_integrated"] = time.time() - t0
timing["Pearth_integrated_calls"] = ncalls

# 3. Other depths, antineutrinos and a latitude-based exposure (not used by GAMBIT, but part of the API)
t0 = time.time()
th12, th13, th23, d, dm21, dm3l = pts[0]
for depth in [0.0, 2000.0]:
    for antinu in [False, True]:
        for iE, E in enumerate([0.8, 9.0]):
            pmns = peanuts.PMNS(th12, th13, th23, d)
            store(f"Pearth_integrated_misc/{depth}/{antinu}/{iE}",
                  peanuts.Pearth_integrated(np.array([0.3, 0.5, 0.2]), earth_density, pmns, dm21, dm3l, E, depth,
                                            antinu=antinu, ns=480, from_file=exposures[0][1], angle="CosZenith"))
for iE, E in enumerate([1.0, 10.0]):
    pmns = peanuts.PMNS(*pts[1][:4])
    store(f"Pearth_integrated_lat/{iE}",
          peanuts.Pearth_integrated(np.array([0.6, 0.3, 0.1]), earth_density, pmns, pts[1][4], pts[1][5], E, 1400.0,
                                    lam=0.7405, ns=60 if full else 30, normalized=False))
timing["misc"] = time.time() - t0

# 4. Pearth at single nadir angles, mass and flavour basis, neutrinos and antineutrinos
t0 = time.time()
etas = np.concatenate([np.linspace(0, np.pi, 41 if full else 13), [np.pi/2, 0.5*np.pi - 1e-9]])
for ip, (th12, th13, th23, d, dm21, dm3l) in enumerate(pts[:2]):
    pmns = peanuts.PMNS(th12, th13, th23, d)
    for iE, E in enumerate([0.3, 5.0, 15.0]):
        for ie, eta in enumerate(etas):
            for depth in [0.0, 1400.0]:
                store_call(f"Pearth/{ip}/{iE}/{ie}/{depth}/mass", Pearth, np.array([0.5, 0.3, 0.2]), earth_density, pmns, dm21, dm3l, E, eta, depth)
                store_call(f"Pearth/{ip}/{iE}/{ie}/{depth}/flav", Pearth, np.array([1.0, 0.0, 0.0], dtype=complex), earth_density, pmns, dm21, dm3l, E, eta, depth, massbasis=False)
                store_call(f"Pearth/{ip}/{iE}/{ie}/{depth}/anti", Pearth, np.array([0.5, 0.3, 0.2]), earth_density, pmns, dm21, dm3l, E, eta, depth, antinu=True)
timing["Pearth"] = time.time() - t0

np.savez(args.output, **results)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()})
print(f"{len(results)} outputs written to {args.output}")
