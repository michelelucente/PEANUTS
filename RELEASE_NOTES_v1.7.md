# PEANUTS v1.7 — release notes (draft, not tagged)

PEANUTS v1.7 reads the B23 / SF-III standard solar models. The physics, the numerical methods and
the results for every solar model supported by v1.6 are unchanged.

## B23 / SF-III solar models

`SolarModel` reads the radial profiles `struct+nu_SF3_<composition>.dat` and the fluxes
`fluxes_SF3_<composition>.dat` of the B23 / SF-III standard solar models (Y. Herrera and
A. Serenelli, ICE-CSIC; GARSTEC models with the Solar Fusion III nuclear rates, Rev. Mod. Phys. 97,
035002 (2025), arXiv:2405.06470), Zenodo version 1.2, DOI 10.5281/zenodo.10822316, licence CC BY 4.0.
All six compositions of the release are supported: GS98, AGSS09, C11, AAG21, MB22m, MB22p.

- The layout is detected from the file name `struct+nu_SF3_<composition>.dat`, or selected with the
  new argument `layout="sf3"` (also `"b16"`, `"bs2005"`, `"bp00"`). In the YAML input of
  `run_peanuts.py` the key is `Solar: layout`.
- The flux file is the `fluxes_SF3_<composition>.dat` in the same directory, unless `flux_file` is
  given. A non-SF3 flux file is an error; a flux file of another composition prints a warning.
- The column numbers (`R_sun`, `nu_<source>`, `log_ne`) are read from the header line of the
  profile file. Units: radius in R_sun, production fractions per unit R_sun normalised to unit
  integral, electron density from log10(n_e / (N_A cm^-3)), the same units as B16; fluxes in
  cm^-2 s^-1.
- The grid has 2001 radii from 0 to 1 R_sun (B16: 1000 radii up to 0.5 R_sun). The cost of one
  `solar_flux_mass` evaluation is proportional to the number of radii: 94 µs for B23 against 49 µs
  for B16 and 60 µs for BS05 (one core of an Apple M2 laptop). The cost of the B16, BS05 and
  BP00 models is unchanged. The table is read as published, over the whole radius: a cut at
  r <= 0.6 R_sun would restore the BS05 cost and change P_ee by at most 1.6e-9, and is not applied.
  One point of the PEANUTS benchmark costs 5 % more than with BS05 for SNO (201 energies, day and
  night) and 10 % more for Borexino.
- `Data/B23_SF3/` ships the GS98 and AGSS09 files, unmodified, with their source, SHA-256 and
  licence in `Data/B23_SF3/README.md`: the high-metallicity reference and its low-metallicity
  counterpart. The other four compositions (C11, AAG21, MB22m, MB22p) are read from the extracted
  Zenodo archive.

Survival probability of electron neutrinos at Δm²₂₁ = 7.5e-5 eV², sin²θ₁₂ = 0.307,
sin²θ₁₃ = 0.022, Δm²₃₁ = 2.5e-3 eV²:

| model | P_ee ⁸B, 10 MeV | P_ee ⁷Be, 0.862 MeV | ⟨P_ee⟩ ⁸B spectrum |
|---|---|---|---|
| BS05(OP) | 0.32485 | 0.52629 | 0.37077 |
| B16-GS98 | 0.32603 | 0.52677 | 0.37251 |
| B23-GS98 | 0.32612 | 0.52680 | 0.37264 |

## Other changes

- `SolarModel` assigns the column names of the profile file in the order of the column numbers.
  pandas assigns the names of `usecols` in file order, so a layout whose density column follows the
  fraction columns (SF3, or explicit columns given out of order) was mislabelled. The B16, BS05 and
  BP00 layouts are already in file order and are read as before.
- `run_peanuts.py` passes the `Solar: fluxscale` key of the YAML input to `SolarModel` (it was read
  but not used).

## Numerical results

For the B16, BS05 and BP00 models shipped in `Data/`, the radius, density and fraction arrays, the
fluxes and the mass-state weights at 4 parameter points × 6 energies × 8 sources are bit-for-bit
identical to v1.6 (`tests/test_solar_models.py`, reference `tests/data/solar_models_v1_6.json`
produced by `tests/exactness/solar_model_reference.py` from the v1.6 tag). The command-line
programs give the same numbers as v1.6 (`PEANUTS_REFERENCE`, `tests/test_cli.py`).

## Tests

`tests/test_solar_models.py` adds the v1.6 regression of the existing models and, for the SF3
layout: detection and explicit selection, column mapping against an independent parse of the file,
normalisation of the production fractions, radius grid, electron-density units, fluxes, mass-state
weight normalisation, and the shifts of the ⁸B P_ee between BS05, B16 and the six B23 compositions
against an independent estimate from the raw tables. With `PEANUTS_SF3_DIR=<SolarModels/ directory
of the Zenodo archive>` the tests cover all six compositions; otherwise GS98 and AGSS09.
