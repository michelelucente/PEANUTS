PEANUTS: Propagation and Evolution of Active NeUTrinoS
========

PEANUTS is a software for the automatic computation of solar neutrino flux and its propagation within Earth. It is designed to be *fast*, so that it can be efficiently used in global scans, and *flexible*, so it allows the input of custom solar models, Earth density profiles, etc.

Detailed documentation about PEANUTS and the related physics computations is available in the companion paper [Eur.Phys.J.C 84 (2024) 2, 119](https://link.springer.com/article/10.1140/epjc/s10052-024-12423-3) [[arXiv:2303.15527 [hep-ph]](https://arxiv.org/abs/2303.15527)].

If you use PEANUTS in your research, please cite:
```bibtex
@article{Gonzalo:2023mdh,
    author = "Gonzalo, Tom{\'a}s E. and Lucente, Michele",
    title = "{PEANUTS: a software for the automatic computation of solar neutrino flux and its propagation within Earth}",
    eprint = "2303.15527",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "TTK-23-001, TTP23-012",
    doi = "10.1140/epjc/s10052-024-12423-3",
    journal = "Eur. Phys. J. C",
    volume = "84",
    number = "2",
    pages = "119",
    year = "2024"
}
```

PEANUTS is distributed under the GPL-3.0 license, which permits the free use and modification of the software for any use. Please read the accompanying LICENSE document for more details of the GPL-3.0 license.

Software requirements
---------------------

PEANUTS runs successfully on any system with a Python 3 environment and the following mandatory packages: numpy, numba, os, copy, time, math, cmath, mpmath, scipy, pyinterval, decimal and pandas. Optional packages for I/O operations are pyyaml and pyslha.

Quick start
-----------

A detailed explanation of how to run PEANUTS and all available options can be found in the reference above. Here are listed the various running modes of PEANUTS without explanation.

The *simple* mode of PEANUTS runs directly on the command line and can be used with

```
run_prob_sun.py [options] <energy> <fraction> [<th12> <th13> <th23> <delta> <dm21> <dm3l>]
```

to compute the probability on the Surface of the Sun, and

```
run_prob_earh.py [options] -f/-m <state> <energy> <eta> <depth> [<th12> <th13> <th23> <delta> <dm21> <dm3l>]
```

to compute the probability at a location below Earth's crust.

The *expert* mode of PEANUTS can be run with

```
run_peanuts.py -f <yaml_file>
```

where the given YAML file determines the specific computation to perform.

Solar models
------------

`SolarModel(solar_model_file, flux_file=None, ..., layout=None)` reads the radial profiles (production fractions of each neutrino source and electron density) and the total fluxes of a standard solar model. The column layout is detected from the file name or selected with `layout`:

| layout | files | shipped in `Data/` |
|---|---|---|
| `"sf3"` | B23 / SF-III (Herrera & Serenelli, [10.5281/zenodo.10822316](https://doi.org/10.5281/zenodo.10822316)): `struct+nu_SF3_<composition>.dat`, fluxes from `fluxes_SF3_<composition>.dat` in the same directory; compositions GS98, AGSS09, C11, AAG21, MB22m, MB22p | `B23_SF3/`: GS98, AGSS09 (source, hashes and licence in `Data/B23_SF3/README.md`) |
| `"b16"` | B16 (Vinyoles et al. 2017): `nudistr_b16_<composition>.dat` with `fluxes_b16.dat` | GS98, AGSS09met (default model: AGSS09met) |
| `"bs2005"` | BS05 (Bahcall, Serenelli & Basu 2005): `bs2005<...>flux.dat` | OP, AGS-OP |
| `"bp00"` | BP00 (Bahcall, Pinsonneault & Basu 2001): `bp00flux.dat` | yes |

For example, `SolarModel("Data/B23_SF3/struct+nu_SF3_GS98.dat")`, or in a YAML file `Solar: {solar_model: Data/B23_SF3/struct+nu_SF3_GS98.dat}` (optionally `layout: sf3`). A file of any other layout is read with the explicit `fluxrows`, `fluxcols`, `fluxscale`, `distrow`, `radiuscol`, `densitycol` and `fractioncols`.

Tests
-----

The test suite in `tests/` checks the physics against independent calculations (matrix exponential, numerical integration, vacuum formula) and the compiled evaluation of `Pearth_integrated`, with its caches, against the direct sum over nadir angles:

```
python3 tests/run_tests.py        # or: pytest tests
```

With `PEANUTS_REFERENCE=<path to another PEANUTS installation>` the command-line programs are also compared with that installation. The performance of the Earth-regeneration path and the regression tests against v1.5 are described in `PERFORMANCE.md`.

Contact
-------

Please contact Michele Lucente (michele.lucente@unibo.it) or Tomas Gonzalo (tomas.gonzalo@kit.edu) for any questions or feedback about PEANUTS.

## Acknowledgements
During the development of part of this project, ML was funded by the European Union under the Horizon Europe's Marie Sklodowska-Curie project 101068791 — NuBridge.
<img width="4125" height="919" alt="EN_FundedbytheEU_RGB_POS" src="https://github.com/user-attachments/assets/00accffd-9843-4f29-8c43-84dc5bd9cf95" />

ML acknowledges as well financial support from the Alexander von Humboldt Foundation during the early stages of the work.

