# B23 / SF-III standard solar models (GS98 and AGSS09 compositions)

Files of the standard solar models computed by Y. Herrera and A. Serenelli (Institute of Space
Sciences, ICE-CSIC, Barcelona) with the GARSTEC code and the nuclear reaction rates of Solar Fusion III
(Acharya et al., Rev. Mod. Phys. 97, 035002 (2025), arXiv:2405.06470).

Source: Y. Herrera and A. Serenelli, "Standard Solar Models B23 / SF-III", Zenodo, version 1.2,
DOI [10.5281/zenodo.10822316](https://doi.org/10.5281/zenodo.10822316), archive
`SolarModels_ICE2023_v1.2.zip` (SHA-256
`382037d76ed35c9b50a4094d3ada3c90284af025c3235b49cda9d932699f87d0`), members
`SolarModels/<file>`, copied unmodified (archive retrieved 2026-11-10).

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>. Work that uses these files cites the Zenodo record
above and the Solar Fusion III review.

| file | content | SHA-256 |
|---|---|---|
| `struct+nu_SF3_GS98.dat` | radial profiles, GS98 composition (Grevesse & Sauval 1998) | `0b89c94221e6b0992f999a96032518ce0941cd65e30ea4b55946c65028ee9d2e` |
| `fluxes_SF3_GS98.dat` | fluxes, fractional errors and correlations, GS98 | `6d7519e07b17180145fd7e670ade9cae0d6dc2382b77ef7da99d57b20369d343` |
| `struct+nu_SF3_AGSS09.dat` | radial profiles, AGSS09 composition (Asplund et al. 2009) | `92e43b7c5fb8206c32c3e1a1a83f2ade1c5abdba5e1b68d0613fb14578206036` |
| `fluxes_SF3_AGSS09.dat` | fluxes, fractional errors and correlations, AGSS09 | `aed56e47be56bb51f68bf9ae0d650e9605aa4b57fdcbb91a5bf1ffb176c97c20` |

The flux files are byte-identical in versions 1.1 (DOI 10.5281/zenodo.10174172) and 1.2; the profile
files differ (version 1.2 revises the profiles; the 8B survival probability changes by less than 1e-5).

The archive also contains the C11, AAG21, MB22m and MB22p compositions with the same layout.
`SolarModel` reads them from the extracted archive: `SolarModel("<dir>/struct+nu_SF3_AAG21.dat")`
finds `fluxes_SF3_AAG21.dat` in the same directory.

## Layout read by `SolarModel` (`layout = "sf3"`)

`struct+nu_SF3_<composition>.dat`: a header line `# R_sun mass_sun L_sun ...`, then 2001 rows,
radius 0 to 1 R_sun in steps of 5e-4 R_sun. Columns used: `R_sun` (0, radius in solar units),
`nu_pp nu_pep nu_hep nu_7Be nu_8B nu_13N nu_15O nu_17F` (9-16, production fractions per unit R_sun,
normalised to unit integral), `log_ne` (17, log10 of the electron density in N_A cm^-3 = mol/cm^3).
The column numbers are taken from the header line. The fractions contain entries of order -1e-11 at
r = 0 and in the outer tail (interpolation noise of the table); they are read as published.

`fluxes_SF3_<composition>.dat`: title line, line of source names, line of fluxes in cm^-2 s^-1
(label `Flux`), fractional errors, 8 x 8 correlation matrix. `SolarModel.flux()` returns the third line.
