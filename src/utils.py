#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

def print_banner():
  """
  Print banner for the program
  """

  banner =  "Propagation and Evolution of Active NeUTrinoS (PEANUTS) \n"\
            "================================\n\n"\
            "Created by:\n"\
            "   Michele Lucente (lucente@physik.rwth-aachen.de)\n"\
            "   Tomas Gonzalo   (tomas.gonzalo@kit.edu)\n\n"\
            "PEANUTS 1.0 is open source and under the terms of the GPL-3 license.\n\n"\
            "Documentation and details for PEANUTS can be found at\n"\
            "T. Gonzalo and M. Lucente, arXiv:xxxx.xxxx\n\n"\
            "==========================================\n"

  print(banner)

def print_inputs(settings):
  """
  Print input values and options
  """

  inputs = ""

  if settings.solar:
    inputs += "\n"\
              "Computing the "
    if settings.probabilities:
      inputs += "probabilities "
      if settings.distorted_spectrum or settings.undistorted_spectrum:
        inputs += "and the "
    if settings.distorted_spectrum:
      inputs += "distorted spectrum "
    elif settings.undistorted_spectrum:
      inputs += "undistorted spectrum "
    inputs += "on the surface of the Sun with values\n\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"\
             "Delta m_{31}^2           : " + str(settings.dm31) + " eV^2\n"\
             "Energy                   : " + str(settings.energy) + " MeV\n"\
             "Neutrino fraction        : " + settings.fraction + "\n"
    if settings.solar_file is not None:
      inputs += "Solar model              : " + settings.solar_file + "\n"

  if settings.earth :
    inputs += "\n"\
              "Computing the probability on Earth with values\n\n"
    if not settings.solar:
      inputs += \
             "Neutrino " + settings.basis + " eigenstate : " + str(settings.nustate) + "\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"\
             "Delta m_{31}^2           : " + str(settings.dm31) + " eV^2\n"\
             "Energy                   : " + str(settings.energy) + " MeV\n"

    if not settings.exposure:
      inputs += \
             "Nadir angle              : " + str(settings.eta) + " rad\n"
    else:
      inputs += \
             "Latitude                 : " + str(settings.latitude) + "\N{DEGREE SIGN}\n"
    inputs += \
             "Depth                    : " + str(settings.depth) + " m\n"\
             "Evolution method         : " + settings.evolution + "\n"
    if settings.density_file is not None:
      inputs += "Earth density              : " + settings.density_file + "\n"


  if not settings.solar and not settings.earth:
    print("Error: Unknown mode.")
    exit()

  print(inputs)


def get_comma_separated_floats(option, opt, value, parser):
  """
  Get an argument from the argparser with
  comma-separated values and make a
  list of floats
  """
  try:
    setattr(parser.values, option.dest, [complex(x) for x in value.split(',')])
  except ValueError:
    print("Error: Wrong format for neutrino state")
    exit()
