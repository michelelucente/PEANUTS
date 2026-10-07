# PEANUTS v1.6 — release notes

PEANUTS v1.6 is a performance release. The public API, the default arguments and the physics are
those of v1.5: no numerical method, grid, tolerance or approximation is changed, and code written for
v1.5 runs unchanged.

## Performance

The analytical Earth-regeneration path (`Pearth_integrated`, `Pearth`, `FullEvolutor`) and the solar
mass-state weights (`solar_flux_mass`) are faster. Cost of one likelihood point of the GAMBIT
NeutrinoBit likelihoods that call PEANUTS, one core of MareNostrum 5 (Xeon Platinum 8480+):

| likelihood | v1.5 | v1.6 | speed-up |
|---|---|---|---|
| SNO (201 energies, day and night) | 5.01 s | 0.69 s | 7.3 |
| Borexino (259 energies × 3 mass states, all nadir angles, ~1 400 solar weights) | 18.18 s | 1.02 s | 17.9 |

Main changes:

- `Pearth_integrated` (analytical mode) evaluates all nadir angles in compiled kernels instead of a
  Python loop over `Pearth`. The nadir exposure is computed once per set of arguments instead of at
  every call (an exposure file is re-read only if it changes). The geometry of the paths is computed
  once per exposure and depth. The spectral decomposition is shared by the paths from above the
  horizon. The projectors of the last call are reused when only the incoming mass-state mixture
  changes, e.g. the three mass states in a row. Exposure samples with zero weight are skipped.
- `Upert` is split into reusable parts. The kinetic Hamiltonian is computed once per energy, `T·T`
  once per shell, and the first-order correction `u1` as a sum of rank-one products. The vanishing
  terms of `u1` are skipped, and the element-wise matrix expressions no longer allocate temporaries.
- `lambdas`, `Iab` and `solar_flux_mass` evaluate their repeated subexpressions once (`th13_M` is no
  longer computed twice).

## Numerical results

Every output differs from v1.5 by at most 6.2e-16 of its scale (2.8e-16 on a probability). The only
source of difference is the rank-one evaluation of `u1`: numba's complex multiplication rounds a
single product differently from the fused multiply-add of the BLAS kernels. These kernels differ
between platforms, so v1.5 itself is reproducible only to this level across machines. All other
changes are bit-for-bit neutral. Details in `PERFORMANCE.md`.

## Tests

New test suite in `tests/`, run with `python3 tests/run_tests.py` or `pytest tests`:

- physics against independent calculations: matrix exponential, numerical integration of the
  Schrödinger equation, quadrature of `Iab`, eigenvalues, PDG vacuum formula, numerical mode;
- the compiled `Pearth_integrated` against the direct sum over nadir angles, for the default,
  custom and tabulated density models;
- cache behaviour: independence from the call history, invalidation when the exposure file
  changes;
- properties: linearity in the state, probability conservation, normalisation of the solar weights;
- errors, fallbacks and the absence of recompilation;
- the command-line programs against a reference installation (`PEANUTS_REFERENCE=<path>`).

`tests/exactness/` contains the regression workload against a previous version (`workload.py`,
`compare.py`, `diff_stats.py`) and the cost benchmark (`bench_point.py`). On MareNostrum 5 all 21
tests pass and nine regression workloads of 7 714 outputs each agree with v1.5 within 6.2e-16 of
the output scale.

## Compatibility

- Requirements unchanged: numpy, numba, scipy, pandas, mpmath; pyyaml and pyslha optional.
- `Pearth_integrated` keeps the v1.5 Python loop as `Pearth_integrated_reference`. It is used
  automatically for `mode="numerical"`, for `full_oscillation=True` and when the state is not a
  one-dimensional numpy array.
- The JIT compilation at the first call of a process is unchanged (about 24 s on MareNostrum 5).

## Known issues, unchanged from v1.5

- `NadirExposure(normalized=True)` calls `scipy.integrate.trapz`, which was removed in SciPy 1.14.
  The examples `SNO_test.yaml` and `exposure_test.yaml`, which set `exposure_normalized: true`,
  therefore fail with recent SciPy.
- The nadir quadrature step is `pi/ns` while the sample spacing is `pi/(ns-1)`, so absolute
  integrals are 0.21 % low; the difference cancels in ratios.
- The switch of `Iab` to its Taylor expansion uses a relative criterion. The closed form can lose
  digits when `|Dl (x2-x1)| <~ 1e-2`.
- `pd.read_csv(delim_whitespace=True)` is deprecated in pandas.
