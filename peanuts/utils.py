#i!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os

def get_git_tag():
  """
  Get the git tag
  """

  try:
    import git
  except:
    print("Warning! Package `gitpython` missing, so could not get git tag. The version displayed might be incorrect.")
    return "v1.2"

  repo = git.Repo(os. getcwd())
  tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
  latest_tag = tags[-1]
  return str(latest_tag)

def print_banner():

  """
  Print banner for the program
  """

  banner =  "Propagation and Evolution of Active NeUTrinoS (PEANUTS) \n"\
            "================================\n\n"\
            "Created by:\n"\
            "   Michele Lucente (michele.lucente@unibo.it)\n"\
            "   Tomas Gonzalo   (tomas.gonzalo@kit.edu)\n\n"\
            "PEANUTS "+get_git_tag()+" is open source and under the terms of the GPL-3 license.\n\n"\
            "Documentation and details for PEANUTS can be found at\n"\
            "T. Gonzalo and M. Lucente, arXiv:2303.15527\n\n"\
            "==========================================\n"

  print(banner)

def print_inputs(settings):
  """
  Print input values and options
  """

  inputs = ""

  if settings.vacuum:
    inputs += "\n"\
              "Computing vacuum oscillation probabilities with values\n\n"
    if not settings.antinu:
        inputs += \
             "Neutrino state           : " + str(settings.nustate) + "\n"
    else:
      inputs += \
             "Antineutrino state       : " + str(settings.nustate) + "\n"
    inputs += \
             "Basis                    : " + settings.basis + "\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"
    if settings.dm3l > 0:
      inputs += \
             "Delta m_{31}^2           : " + str(settings.dm3l) + " eV^2\n"
    else:
      inputs += \
             "Delta m_{32}^2           : " + str(settings.dm3l) + " eV^2\n"
    inputs += \
             "Energy                   : " + str(settings.energy) + " MeV\n"\
             "Baseline                 : " + str(settings.baseline) + " km\n"

  if settings.solar:
    inputs += "\n"\
              "Computing the "
    if settings.solar_probabilities:
      inputs += "probabilities "
      if settings.distorted_spectrum or settings.undistorted_spectrum:
        inputs += "and the "
    if settings.distorted_spectrum:
      inputs += "distorted spectrum "
    elif settings.undistorted_spectrum:
      inputs += "undistorted spectrum "
    if not settings.solar_probabilities and not solar.distorted_spectrum and not solar.undistorted_spectrum:
      inputs += "oscillations "
    inputs += "on the surface of the Sun with values\n\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"
    if settings.dm3l > 0:
      inputs += \
             "Delta m_{31}^2           : " + str(settings.dm3l) + " eV^2\n"
    else:
      inputs += \
             "Delta m_{32}^2           : " + str(settings.dm3l) + " eV^2\n"
    inputs += \
             "Energy                   : " + str(settings.energy) + " MeV\n"\
             "Neutrino fraction        : " + settings.fraction + "\n"
    if settings.solar_file is not None:
      inputs += \
             "Solar model              : " + settings.solar_file + "\n"
    inputs += \
             "Density profile          : " + ("adiabatic" if settings.adiabatic else "exponential") + "\n"

  if settings.atmosphere:
    inputs += "\n"\
              "Computing the "
    if settings.atm_probabilities:
      inputs += "probabilities "
      if settings.atm_evolved_state:
        inputs += "and the "
    if settings.atm_evolved_state:
      inputs += "evolved state "
    if not settings.atm_probabilities and not settings.atm_evolved_state:
      inputs += "oscillations "
    inputs += "on the surface of the Earth of neutrinos traversing the atmosphere, with values\n\n"
    if not settings.solar:
       if not settings.antinu:
         inputs += \
             "Neutrino state           : " + str(settings.nustate) + "\n"
       else:
         inputs += \
             "Antineutrino state       : " + str(settings.nustate) + "\n"
       inputs += \
             "Basis                    : " + settings.basis + "\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"
       if settings.dm3l > 0:
         inputs += \
             "Delta m_{31}^2           : " + str(settings.dm3l) + " eV^2\n"
       else:
         inputs += \
             "Delta m_{32}^2           : " + str(settings.dm3l) + " eV^2\n"
       inputs += \
             "Energy                   : " + str(settings.energy) + " MeV\n"\
             "Height                   : " + str(settings.height) + " m\n"
    else:
       inputs += \
             "Height (max)             : " + str(settings.height) + " m\n"

    if not settings.earth or not settings.exposure:
      inputs += \
             "Nadir angle              : " + str(settings.eta) + " rad\n"




  if settings.earth :
    inputs += "\n"\
              "Computing the probability through the Earth with values\n\n"
    if not settings.solar and not settings.atmosphere:
      if not settings.antinu:
        inputs += \
             "Neutrino state           : " + str(settings.nustate) + "\n"
      else:
        inputs += \
             "Antineutrino state       : " + str(settings.nustate) + "\n"
      inputs += \
             "Basis                    : " + settings.basis + "\n"\
             "theta_{12}               : " + str(settings.theta12) + "\n"\
             "theta_{13}               : " + str(settings.theta13) + "\n"\
             "theta_{23}               : " + str(settings.theta23) + "\n"\
             "delta_CP                 : " + str(settings.delta) + "\n"\
             "Delta m_{21}^2           : " + str(settings.dm21) + " eV^2\n"
      if settings.dm3l > 0:
        inputs += \
             "Delta m_{31}^2           : " + str(settings.dm3l) + " eV^2\n"
      else:
        inputs += \
             "Delta m_{32}^2           : " + str(settings.dm3l) + " eV^2\n"
      inputs += \
             "Energy                   : " + str(settings.energy) + " MeV\n"

    if not settings.atmosphere and not settings.exposure:
      inputs += \
             "Nadir angle              : " + str(settings.eta) + " rad\n"
    elif not settings.atmosphere:
      if settings.latitude != -1:
        inputs += \
             "Latitude                 : " + str(settings.latitude) + "\N{DEGREE SIGN}\n"
      inputs += \
             "Exposure normalized      : " + str(settings.exposure_normalized) + "\n"\
             "Exposure time            : " + str(settings.exposure_time) + "\n"\
             "Exposure samples         : " + str(settings.exposure_samples) + "\n"
      if settings.exposure_file is not None:
        inputs += \
             "Exposure file            : " + settings.exposure_file + "\n"\
             "Exposure angle           : " + settings.exposure_angle + "\n"

    inputs += \
             "Depth                    : " + str(settings.depth) + "  m\n"
    inputs += \
             "Evolution method         : " + settings.evolution + "\n"
    if settings.density_file is not None:
      inputs += \
             "Earth density            : " + settings.density_file + "\n"


  if not settings.vacuum and not settings.solar and not settings.atmosphere and not settings.earth:
    print("Error: Unknown mode.")
    exit()

  print(inputs)


def get_comma_separated_floats(option, opt, value, parser):
  """
  Get an argument from the argparser with
  comma-separated values and make a
  list of complex numbers
  """
  try:
    setattr(parser.values, option.dest, [complex(x) for x in value.split(',')])
  except ValueError:
    print("Error: Wrong format for neutrino state")
    exit()
