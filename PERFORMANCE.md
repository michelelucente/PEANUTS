# Performance of the Earth-regeneration path: audit and speed-up

Branch `perf/exact-speedup`, based on v1.5 (the version GAMBIT downloads). No numerical method,
grid, tolerance or approximation is changed and the public API and its defaults are unchanged.
All changes but one remove redundant work (an expression evaluated once instead of several times,
with the same operands and the same BLAS call) and are bit-for-bit neutral. The exception is the
evaluation of the first-order correction `u1` as a rank-one product, which changes the rounding of
single complex products: the results differ from v1.5 by at most 6.2e-16 of the output scale (see
[Regression against v1.5](#regression-against-v15)).

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

| file | change | why the result is unchanged |
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
| `evolutor.py: evolutor_from_spectrum` | each term `M_a diag(d,0,0) M_b` of `u1` is the outer product `(M_a[:,0] d) M_b[0,:]`, evaluated directly instead of with two BLAS products: 12 of the 13 matrix products of each `Upert` call removed | not bit-for-bit: see below |

The rank-one evaluation computes each element of a `u1` term as the same single complex product
`x y` as the BLAS product, but with numba's scalar arithmetic, which rounds differently
(`tests/exactness/probe_complex_product.py`, 20 000 random products, `rn` = rounding to double):

| arithmetic | real part | imaginary part |
|---|---|---|
| numba scalar | `rn(rn(xr yr) - rn(xi yi))` | `rn(rn(xr yi) + rn(xi yr))` |
| OpenBLAS 0.3.27, Xeon 8480+ (MareNostrum 5) | `rn(xr yr - rn(xi yi))` | `rn(xr yi + rn(xi yr))` |
| OpenBLAS 0.3.30, Apple M2 | `rn(rn(xr yr) - xi yi)` | `rn(xr yi + rn(xi yr))` |

The kernels use a fused multiply-add, so one of the two partial products is not rounded; the result
differs from the scalar product in 44 % of the cases, by at most 2.2e-16 relative. The table also
shows that the BLAS libraries of the two platforms round the real part differently, so the last bits
of the v1.5 results themselves depend on the platform.

## Regression against v1.5

`tests/exactness/workload.py` evaluates the entry points with the call pattern of the GAMBIT
frontend (`Backends/src/frontends/PEANUTS_1_5.cpp`) and stores every output; `compare.py` compares
two runs, byte by byte or within a tolerance relative to the largest value of each output. One
workload covers 12 parameter points (4 fixed, 8 drawn from a seed; normal and inverted ordering,
`theta13 = 0`, `dm21` from 3e-6 to 3e-4 eV^2), 25 energies × 8 sources of `solar_flux_mass`,
`Psolar`, 3 456 `Pearth_integrated` calls (SNO and LNGS exposures, day, night and full, mass-state
mixtures and the three mass states in a row), depths 0 and 2000 m, antineutrinos, a
latitude-based exposure, and `Pearth` at 43 nadir angles in the mass and flavour bases, including
the exceptions v1.5 raises.

| platform | workloads | outputs per workload | result |
|---|---|---|---|
| MareNostrum 5 GPP (Xeon Platinum 8480+), numpy 2.1.1, numba 0.62.1, scipy 1.14.1, scipy-openblas 0.3.27 (the Python and BLAS of the GAMBIT runs; reference: the installed GAMBIT backend `peanuts/1.5`) | 9 seeds | 7 714 | 61-107 outputs differ per workload; largest difference 6.2e-16 of the output scale (2.8e-16 on a probability), median 0.8-6e-17; none above 1e-15 |
| Apple M2, numpy 2.2.6, numba 0.61.2, scipy 1.16.0, OpenBLAS 0.3.30 | 1 (small) | 1 186 | 6 differ, largest 2.6e-16 of the output scale |

The largest differences relative to the value itself, up to 1.8e-2, are on `P(nu_3 -> nu_e)` at
`theta13 = 0`, which vanishes analytically and is evaluated as about 6e-34 (rounding residue, absolute
difference 1e-35). Without the rank-one evaluation (commit f1404e9) every output is bit-for-bit
identical to v1.5 on both platforms.

## Test suite

`tests/` (run with `python3 tests/run_tests.py` or `pytest tests`) checks the new code against
independent calculations and checks the compiled path against the direct sum. On MareNostrum 5 all
21 tests pass (job of 2026-10-06).

| test | reference | tolerance (measured) |
|---|---|---|
| constant-density evolutor | `scipy.linalg.expm(-i H L)`, 400 random cases | 2e-13 per radian of phase (4e-14) |
| `lambdas` | `numpy.linalg.eigvalsh(T)` | 1e-10 of the spectrum scale |
| `Iab` | numerical quadrature | 1e-10 (analytic branch, `|Dl (x2-x1)| >= 0.1`), 1e-6 (Taylor branch) |
| first-order evolutor of the Earth shells | DOP853 integration, `rtol = 1e-11` | 2e-3 and at least 3 times better than zeroth order (7e-4, 4.8) |
| vacuum probabilities | PDG formula written independently | 1e-13 per radian of phase |
| depth 0, path from above the horizon | `|U|^2 w` | 1e-15 |
| analytical against numerical mode of PEANUTS | `Pearth(mode="numerical")` | 2e-3 (9e-4) |
| compiled `Pearth_integrated` | direct sum `Pearth_integrated_reference`, three density models (default, custom, tabulated), depths 0, 1400, 2092 m, day/night/full, neutrinos and antineutrinos, file and latitude exposures | bit-for-bit |
| history independence | 150 calls in random order revisiting parameters, energies, depths, density models and states | bit-for-bit against fresh direct sums |
| cache invalidation | exposure file rewritten between calls | bit-for-bit against the new file |
| linearity in the state | `P(w) = sum_j w_j P(e_j)` | 1e-14 |
| probability conservation | exposure integral | 1e-13 (day), 5e-3 (night: first-order unitarity defect, the same as v1.5, 2e-3 at `dm21 = 1e-5` eV^2 and 15 MeV) |
| errors and fallbacks | v1.5 behaviour: `ValueError` for `eta` outside `[0, pi]`, exit for a state of size != 3 or `ns` != file samples, list states | same behaviour |
| recompilation | numba signatures of the kernels after new points, orderings, neutrino types | none added |
| solar weights | normalisation, vacuum limit `|U_ei|^2`, `th12_M_given_th13_M == th12_M`, `Psolar` normalisation | 1e-13, 1e-14, bit-for-bit, 1e-13 |
| command-line programs | the same commands run with `PEANUTS_REFERENCE` (v1.5): `run_peanuts.py` on every example, `run_prob_sun.py`, `run_prob_earth.py` | 1e-12 relative on every printed number, or the same error |

## Measured cost per likelihood point

`tests/exactness/bench_point.py` reproduces one SNO and one Borexino point with the frontend's
calls; the first point, which carries the JIT compilation, is excluded.

| case (one core, full scale) | v1.5 | this branch | ratio |
|---|---|---|---|
| SNO, MareNostrum 5 | 5.01 s | 0.69 s | 7.3 |
| Borexino, MareNostrum 5 | 18.18 s | 1.02 s | 17.9 |

The largest difference of the benchmark outputs from v1.5 is 2.1e-16 of the output scale. The cost
is linear in the number of energies. The JIT compilation of a process takes about 24 s on
MareNostrum 5 (27 s with v1.5), once per process. Without the rank-one evaluation (bit-for-bit
identical to v1.5) the costs are 1.11 s (SNO, ratio 4.5) and 1.41 s (Borexino, ratio 12.9).

## Pre-existing issues found, not changed

| issue | where | consequence |
|---|---|---|
| `NadirExposure(normalized=True)` calls `scipy.integrate.trapz`, removed in SciPy 1.14 (MareNostrum 5 runs 1.14.1); with an exposure file `exposure` is a list, which cannot be divided by `norm` | `time_average.py: NadirExposure` | normalised exposures fail; GAMBIT passes `normalized=false` |
| the quadrature step is `pi/ns` while the nadir grid is `linspace(0, pi, ns)`, step `pi/(ns-1)` | `earth.py: Pearth_integrated`, `time_average.py: NadirExposure` | absolute integrals are 0.21 % low; cancels in ratios (SNO divides by the total, Borexino normalises each column) |
| `Pearth` raises `ZeroDivisionError` for a grazing path, e.g. depth 0 and `eta = pi/2 - 1e-9`, where the shell chord vanishes | `evolutor.py: average_density` | not reached by the 480-sample grids (nearest sample 3.3e-3 rad from `pi/2`) |
| `Iab` switches to its Taylor expansion when `|Dl/(la+lb)| < 1e-2`, a relative criterion; when `|Dl (x2-x1)| <~ 1e-2` but the relative criterion fails, the closed form (terms in `Dl^-5`) loses digits | `integration.py: Iab` | up to 3e-5 of the integral's scale in random probes, 1e-15 for `|Dl (x2-x1)| >= 1`; enters the first-order correction only |
| the examples `SNO_test.yaml` and `exposure_test.yaml` set `exposure_normalized: true` | `examples/` | they fail with SciPy >= 1.14 (first row) |
| `pd.read_csv(delim_whitespace=True)` is deprecated | `solar.py: SolarModel` | `FutureWarning` now, error in a future pandas |
| `cache=True` cannot be used for the functions that take the `PMNS` and `EarthDensity` jitclasses | numba | JIT compilation (24-27 s per process on MareNostrum 5) is repeated in every process |

The GAMBIT frontend `PEANUTS_1_5.cpp` has two defects of its own: `Pearth` passes the keyword
`basis`, which `peanuts.Pearth` does not have (`massbasis`), so that backend function raises
`TypeError` if called; `Pearth_integrated` builds a `py::dict options` it never uses. No
frontend change is needed for the speed-up.

## Running the tests

```
python3 tests/run_tests.py                                   # test suite (or: pytest tests)
PEANUTS_REFERENCE=<v1.5 root> python3 tests/run_tests.py     # also compares the command-line programs
python3 tests/exactness/workload.py <v1.5 root> ref.npz --size full [--seed S] --exposure-dir <dir with LNGSCosZenith.dat>
python3 tests/exactness/workload.py . new.npz --size full [--seed S] --exposure-dir <same dir>
python3 tests/exactness/compare.py ref.npz new.npz --scale-tol 1e-14
python3 tests/exactness/diff_stats.py ref.npz new.npz
python3 tests/exactness/ast_equivalence.py <v1.5 root>/peanuts peanuts integration:lambdas:w,w23,w13 integration:Iab:eDl
python3 tests/exactness/bench_point.py <root> sno|borexino [--scale F] --exposure-dir <dir>
python3 tests/exactness/probe_complex_product.py
```
