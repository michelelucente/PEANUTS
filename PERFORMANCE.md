# Performance of the Earth-regeneration path: audit and exact speed-up

Branch `perf/exact-speedup`, based on v1.5 (the version GAMBIT downloads). Every change removes
redundant work: an expression is evaluated once instead of several times, with the same operands
and, for matrix products, the same BLAS call. No numerical method, grid, tolerance or
approximation is changed, and the outputs are bit-for-bit identical to v1.5 (see
[Exactness](#exactness)). The public API and its defaults are unchanged.

## Where the time goes in a GAMBIT likelihood

A single probability evaluation of v1.5 (one energy, one nadir angle, `Pearth`) costs 36 µs for a
path that crosses the Earth and 6 µs for a path from above the horizon (Apple M2, one core), in
line with the CPU times of arXiv:2303.15527, Fig. 9. The cost of a likelihood point is the number
of such evaluations it requests:

| likelihood (NeutrinoBit) | calls per point | `Pearth` evaluations per point |
|---|---|---|
| SNO (`Oscillations_SNO.cpp`, via `nu_osc_prob_peanuts`) | 402 `Pearth_integrated` (201 energies × day, night) + 402 `solar_flux_mass` | 96 480 (85 626 with non-zero exposure) |
| Borexino (`Oscillations_Borexino.cpp: node_probabilities`) | 777 `Pearth_integrated` (259 energies × 3 mass states, all nadir angles) + ~1 400 `solar_flux_mass` | 372 960 |

In v1.5, besides the evaluations themselves, three kinds of work are repeated within a point:

1. `Pearth_integrated` recomputes the nadir exposure (`NadirExposure`: file read, cubic
   interpolation of 480 samples) at every call, although it depends on the experiment only:
   4.4 ms per call, about 30 % of a Borexino point.
2. The probability is linear in the incoming mass-state mixture, so the three calls that
   Borexino makes for the three mass states at the same energy compute the same evolutors three
   times.
3. Inside an evaluation: the kinetic Hamiltonian `U diag(k) U^T` is rebuilt for every shell; for
   paths from above the horizon every quantity except the path length depends on the energy and
   on the average density only, and that density takes a handful of distinct values over the
   nadir grid; `T·T` is recomputed three times per shell; the vanishing diagonal terms of the
   first-order correction `u1` are computed (two matrix products each); `th13_M` is computed twice
   per call of `solar_flux_mass`; `lambdas` and `Iab` evaluate the same complex powers, square
   roots and exponentials up to six times; every element-wise matrix expression and every
   `np.dot` allocates a temporary array.

## Changes

| file | change | exact because |
|---|---|---|
| `earth.py: Pearth_integrated` | analytical mode: compiled kernels over all nadir samples (`path_geometry`, `mass_projectors`, `integrate_projectors`); the direct Python loop is kept as `Pearth_integrated_reference` and is used for the numerical mode and `full_oscillation` | same sum `prob += P(eta) * w(eta) * deta`, same order, same association |
| `earth.py: cached_exposure` | the exposure is computed once per set of arguments (file identified by path, size and modification time) | same function, same inputs |
| `earth.py: path_geometry` | shell parameters and path lengths of every nadir angle computed once per exposure and depth, shared by all energies | energy-independent; same expressions |
| `earth.py: mass_projectors` | for paths from above the horizon, the spectral decomposition is shared by all angles whose average density is equal bit by bit | the decomposition is a function of that density only |
| `earth.py: integrated_projectors` | the matrices `|S U|^2` of the last call are kept; a call with the same parameters, energy, depth and exposure and a different mass-state mixture reuses them | `Pearth_analytical` computes `real(|S U|^2 · nustate)` with these same matrices |
| `earth.py` | samples with zero exposure weight are not evaluated | they add `P * 0 * deta = 0`; differs only if `P` is not finite |
| `evolutor.py` | `Upert` split into `kinetic_terms`, `kinetic_hamiltonian`, `average_density`, `spectral_decomposition`, `evolutor_from_spectrum`; `FullEvolutor` built from `evolutor_setup` and `crossing_evolutor`, which take the energy-dependent quantities once per evaluation | same operations on the same operands |
| `evolutor.py` | `T·T` computed once per shell; diagonal terms of `u1` skipped | `Iab` returns 0 for `la == lb`: the skipped terms are zero matrices |
| `evolutor.py` | element-wise matrix expressions written as loops; `np.dot(a, b, out)` into preallocated buffers | numba implements `np.dot(a, b)` as `np.dot(a, b, np.empty(...))`; the loops perform the same scalar operations |
| `integration.py: lambdas, Iab` | repeated subexpressions evaluated once | checked by `tests/exactness/ast_equivalence.py`: substituting the new names back gives the v1.5 syntax tree |
| `matter_mixing.py`, `solar.py: Tei` | `th12_M_given_th13_M` reuses the `th13_M` already computed | same function, same inputs |

Option, off by default: `PEANUTS_RANK1_U1=1` evaluates each term of `u1 = Σ M_a diag(d,0,0) M_b`
as the rank-one product `(M_a[i,0] d) M_b[0,j]` instead of two BLAS products. It removes 12 of the
13 matrix products of each `Upert` call, but it equals the BLAS result only if the BLAS kernel
rounds a single complex product like the scalar complex multiplication: it does not with
OpenBLAS 0.3.30 on ARM (6 of 1 186 outputs differ, by at most 4.2e-16 relative on an O(1)
probability). It must not be used on a platform where `tests/exactness` has not passed with it.

## Exactness

`tests/exactness/workload.py` evaluates the entry points with the call pattern of the GAMBIT
frontend (`Backends/src/frontends/PEANUTS_1_5.cpp`) and stores every output; `compare.py` compares
two runs byte by byte. The full workload covers 12 parameter points (normal and inverted ordering,
`theta13 = 0`, `dm21` from 3e-6 to 3e-4 eV^2), 25 energies × 8 sources of `solar_flux_mass`,
`Psolar`, 3 456 `Pearth_integrated` calls (SNO and LNGS exposures, day, night and full, mass-state
mixtures and the three mass states in a row), depths 0 and 2000 m, antineutrinos, a
latitude-based exposure, and `Pearth` at 43 nadir angles in the mass and flavour bases, including
the exceptions v1.5 raises.

| platform | outputs | bit-for-bit identical |
|---|---|---|
| Apple M2, numpy 2.2.6, numba 0.61.2, scipy 1.16.0, OpenBLAS 0.3.30 | 7 714 | 7 714 |
| MareNostrum 5 GPP, production Python | pending | pending |

`bench_point.py` (below) also compares its outputs: identical on every run.

## Measured cost per likelihood point

`tests/exactness/bench_point.py` reproduces one SNO and one Borexino point with the frontend's
calls, excluding the first point (JIT compilation, about 10 s per process, unchanged).

| case (Apple M2, one core) | v1.5 | this branch | ratio |
|---|---|---|---|
| SNO, 10 % of the energies | 0.254 s | 0.061 s | 4.2 |
| Borexino, 10 % of the energies and nodes | 1.077 s | 0.105 s | 10.3 |

The cost is linear in the number of energies, so a full point is about ten times these figures.
In a Borexino point the remaining time is, estimated from the cost of the components, ~85 % paths
that cross the Earth (~15 µs per nadir angle, ~3 µs per `Upert` call), ~7 % paths from above the
horizon (~1 µs per nadir angle) and ~6 % `solar_flux_mass` (0.052 ms per call, 0.087 ms in v1.5).

## Pre-existing issues found, not changed

| issue | where | consequence |
|---|---|---|
| `NadirExposure(normalized=True)` calls `scipy.integrate.trapz`, removed in SciPy 1.14 (MareNostrum 5 runs 1.14.1); with an exposure file `exposure` is a list, which cannot be divided by `norm` | `time_average.py: NadirExposure` | normalised exposures fail; GAMBIT passes `normalized=false` |
| the quadrature step is `pi/ns` while the nadir grid is `linspace(0, pi, ns)`, step `pi/(ns-1)` | `earth.py: Pearth_integrated`, `time_average.py: NadirExposure` | absolute integrals are 0.21 % low; cancels in ratios (SNO divides by the total, Borexino normalises each column) |
| `Pearth` raises `ZeroDivisionError` for a grazing path, e.g. depth 0 and `eta = pi/2 - 1e-9`, where the shell chord vanishes | `evolutor.py: average_density` | not reached by the 480-sample grids (nearest sample 3.3e-3 rad from `pi/2`) |
| `pd.read_csv(delim_whitespace=True)` is deprecated | `solar.py: SolarModel` | `FutureWarning` now, error in a future pandas |
| `cache=True` cannot be used for the functions that take the `PMNS` and `EarthDensity` jitclasses | numba | JIT compilation (~10 s locally, 35-50 s per rank on a cluster) is repeated in every process |

The GAMBIT frontend `PEANUTS_1_5.cpp` has two defects of its own: `Pearth` passes the keyword
`basis`, which `peanuts.Pearth` does not have (`massbasis`), so that backend function raises
`TypeError` if called; `Pearth_integrated` builds a `py::dict options` it never uses. No
frontend change is needed for the speed-up.

## Running the tests

```
python3 tests/exactness/workload.py <v1.5 root> ref.npz --size full --exposure-dir <dir with LNGSCosZenith.dat>
python3 tests/exactness/workload.py . new.npz --size full --exposure-dir <same dir>
python3 tests/exactness/compare.py ref.npz new.npz
python3 tests/exactness/ast_equivalence.py <v1.5 root>/peanuts peanuts integration:lambdas:w,w23,w13 integration:Iab:eDl
python3 tests/exactness/bench_point.py <root> sno|borexino [--scale F] --exposure-dir <dir>
```
