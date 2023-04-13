PEANUTS: Propagation and Evolution of Active NeUTrinoS
========

PEANUTS is a software for the automatic computation of solar neutrino flux and its propagation within Earth. It is designed to be *fast*, so that it can be efficiently used in global scans, and *flexible*, so it allows the input of custom solar models, Earth density profiles, etc.

Please cite the following paper when using PEANUTS:

T. E. Gonzalo and M. Lucente, arXiv:2303.15527

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

Contact
-------

Please contact Michele Lucente (michele.lucente@unibo.it) or Tomas Gonzalo (tomas.gonzalo@kit.edu) for any questions or feedback about PEANUTS.
