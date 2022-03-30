#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

def print_banner():
  """
  Print banner for the program
  """

  banner =  "Solar Neutrino Fluxes (SNuF) \n"\
            "================================\n\n"\
            "Created by:\n"\
            "   Michele Lucente (lucente@physik.rwth-aachen.de)\n"\
            "   Tomas Gonzalo   (gonzalo@physik.rwht-aachen.de)\n\n"\
            "SNuF 1.0 is open source and under the terms of the GPL-3 license.\n\n"\
            "Documentation and details for SNuF can be found at\n"\
            "T. Gonzalo and M. Lucente, arXiv:xxxx.xxxx\n\n"\
            "==========================================\n"

  print(banner)

def print_inputs(mode, options, pmns, md21, md31, E, etaorfraction, H=0):
  """
  Print input values and options
  """

  if mode == "earth" :
    inputs = "Computing the probability on Earth with values\n\n"\
             "Neutrino " + ("flav" if options.flavour != None else "mass") + " eigenstate : " + (str(options.flavour) if options.flavour != None else str(options.mass)) + "\n"\
             "theta12                  : " + str(pmns.theta12) + "\n"\
             "theta13                  : " + str(pmns.theta13) + "\n"\
             "theta23                  : " + str(pmns.theta23) + "\n"\
             "delta                    : " + str(pmns.delta) + "\n"\
             "Delta m_{21}^2           : " + str(md21) + " eV^2\n"\
             "Delta m_{31}^2           : " + str(md31) + " eV^2\n"\
             "Energy                   : " + str(E) + " MeV\n"\
             "Nadir angle              : " + str(etaorfraction) + " rad\n"\
             "Depth                    : " + str(H) + " m\n"\
             "Evolution method         : " + ("analytical" if options.analytical else "numerical") + "\n"

  elif mode == "solar":
    inputs = "Computing the probability on the surface of the Sun with values\n\n"\
             "theta12                  : " + str(pmns.theta12) + "\n"\
             "theta13                  : " + str(pmns.theta13) + "\n"\
             "theta23                  : " + str(pmns.theta23) + "\n"\
             "delta                    : " + str(pmns.delta) + "\n"\
             "Delta m_{21}^2           : " + str(md21) + " eV^2\n"\
             "Delta m_{31}^2           : " + str(md31) + " eV^2\n"\
             "Energy                   : " + str(E) + " MeV\n"\
             "Neutrino fraction        : " + etaorfraction + "\n"

  else:
    print("Error: Unknown mode.")

  print(inputs)


def get_comma_separated_floats(option, opt, value, parser):
  """
  Get an argument from the argparser with
  comma-separated values and make a
  list of floats
  """
  setattr(parser.values, option.dest, [float(x) for x in value.split(',')])

