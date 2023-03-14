#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np

with_slha = True
try:
  import pyslha # pyslha module for reading SLHA files, by Andy Buckley (https://arxiv.org/abs/1305.4194)
except:
  print("Warning!: Python module pyslha not found, disabling slha reading routines")
  with_slha = False

with_yaml = True
try:
  import yaml
except:
  print("Warning!: Python module pyyaml not found, disabling yaml reading routines")
  with_yaml = False

import pandas as pd
from decimal import Decimal

from peanuts.settings import Settings
import os


def read_csv(*args, **kwargs):
  """
  Function to read csv. Just a wrapper around the pandas version
  """
  return pd.read_csv(*args, **kwargs)

def read_yaml(filepath):
  """
  Function to read from yaml files
  """

  if not with_yaml:
    print("Error!: Tried to read a yaml file, but module pyyaml is not installed")
    exit()

  settings = Settings(yaml.safe_load(open(filepath)))

  return settings

def read_slha(filepath):
  """
  Function to read from slha files
  """

  if not with_slha:
    print("Error!: Tried to use pyslha but module is not installed")
    exit()


  slha = pyslha.read(filepath)
  blocks = [''.join(k) for k in slha.blocks.keys()]

  # Check that the slha contains the important blocks
  if 'SMINPUTS' not in blocks:
    raise Exception("Error: SLHA file has the wrong format, the block SMINPUTS in missing")
  if 'UPMNSIN' not in blocks:
    raise Exception("Error: SLHA file has the wrong format, the block UPMNS in missing")

  # Extract masses from the SMINPUTS block (converting from GeV to eV)
  mNu1 = slha.blocks['SMINPUTS'][12] * 1e9
  mNu2 = slha.blocks['SMINPUTS'][14] * 1e9
  mNu3 = slha.blocks['SMINPUTS'][8] * 1e9

  # Extract PMNS angles and phases from the UPMNSIN block
  theta12 = slha.blocks['UPMNSIN'][1]
  theta23 = slha.blocks['UPMNSIN'][2]
  theta13 = slha.blocks['UPMNSIN'][3]
  delta = slha.blocks['UPMNSIN'][4]
  # TODO: I don't think these are useful in any way, so commented for now
  #alpha1 = slha.blocks['UMPNSIN'][5]
  #alpha2 = slha.blocks['UMPNSIN'][6]

  # Build dictionary of neutrino parameters
  nu_params = {'dm21' : mNu2**2 - mNu1**2,
               'dm3l' : mNu3**2 - mNu1**2 if mNu3 > mNu1 else mNu3**2 - mNu2**2,
               'theta12'    : theta12,
               'theta23'    : theta23,
               'theta13'    : theta13,
               'delta'      : delta}

  return nu_params

def output(settings, outs):

  def dec(x):
    return "{:.5E}".format(Decimal(x))

  towrite = ""

  if settings.solar and settings.flux:
    towrite += "# Flux [cm^-2 s^-1]\t" + str(dec(outs[0]["flux"])) + "\n"

  if settings.probabilities:
    towrite += "\n# Probabilities\n# "
    if "energy" in settings.scan.labels:
      towrite += "E [MeV]\t"
    if "eta" in settings.scan.labels:
      towrite += "Nadir angle\t"
    if "theta12" in settings.scan.labels:
      towrite += "theta_{12}\t"
    if "theta13" in settings.scan.labels:
      towrite += "theta_{13}\t"
    if "theta23" in settings.scan.labels:
      towrite += "theta_{23}\t"
    if "delta" in settings.scan.labels:
      towrite += "delta_CP\t"
    if "dm21" in settings.scan.labels:
      towrite += "Dm21^2 [eV^2]\t"
    if "dm3l" in settings.scan.labels:
      towrite += "Dm3l^2 [eV^2]\t"

    if settings.solar:
      if not settings.antinu:
        towrite += "Psolar (e) \t Psolar (mu) \t Psolar (tau)\t"
      else:
        towrite += "Psolar (~e) \t Psolar (~mu) \t Psolar (~tau)\t"
    if settings.earth:
      if not settings.antinu:
        towrite += "Pearth (e) \t Pearth (mu) \t Pearth (tau)\t"
      else:
        towrite += "Pearth (~e) \t Pearth (~mu) \t Pearth (~tau)\t"
      if settings.evolved_state:
        towrite += "Evolved " + ("anti" if settings.antinu else "") + "neutrino state\t"
    towrite += "\n"

    for i, param in settings.scan.enumerate():
      if "energy" in settings.scan.labels:
        towrite += str(dec(param.energy)) + "\t"
      if "eta" in settings.scan.labels:
        towrite += str(dec(param.eta)) + "\t"
      if "theta12" in settings.scan.labels:
        towrite += str(dec(param.theta12)) + "\t"
      if "theta13" in settings.scan.labels:
        towrite += str(dec(param.theta13)) + "\t"
      if "theta23" in settings.scan.labels:
        towrite += str(dec(param.theta23)) + "\t"
      if "delta" in settings.scan.labels:
        towrite += str(dec(param.delta)) + "\t"
      if "dm21" in settings.scan.labels:
        towrite += str(dec(param.dm21)) + "\t"
      if "dm3l" in settings.scan.labels:
        towrite += str(dec(param.dm3l)) + "\t"

      if settings.solar:

        for out in outs[i]["solar"]:
          towrite += str(dec(out)) + "\t"

      if settings.earth:
        for out in outs[i]["earth"]:
          towrite += str(dec(out)) + "\t"
        if settings.evolved_state:
          towrite += str([np.around(out,5) for out in outs[i]["evolved_state"]]) + "\t"

      towrite += "\n"

  if settings.solar and settings.undistorted_spectrum:
    towrite += "\n# Spectrum (undistorted)\n"
    towrite += "# E [MeV] \t Spec (e)\n"

    if "energy" in settings.scan.labels:
      for i, param in settings.scan.enumerate():
        towrite += str(dec(param.energy)) + "\t"
        towrite += str(dec(outs[i]['spectrum']))
    else:
      towrite += str(dec(settings.energy)) + "\t"
      towrite += str(dec(outs[0]['spectrum']))

    towrite += "\n"


  elif settings.solar and settings.distorted_spectrum:
    towrite += "\n# Spectrum (distorted)\n"
    towrite += "# E [MeV] \t Spec (e) \t Spec (mu) \t Spec (tau)\n"

    if "energy" in settings.scan.labels:
      for i, param in settings.scan.enumerate():
        towrite += str(dec(param.energy)) + "\t"
        for out in outs[i]["spectrum"]:
          towrite += str(dec(out)) + "\t"
    else:
      towrite += str(dec(settings.energy)) + "\t"
      for out in outs[0]["spectrum"]:
        towrite += str(dec(out)) + "\t"

    towrite += "\n"


  if settings.output == "stdout":

    print()
    print(towrite)

  else:

    try:

      path = os.path.dirname(os.path.realpath( __file__ ))
      f = open(path + "/../" + settings.output, "w")
      f.write(towrite)
      f.close()

      print("Output written to file ", settings.output)

    except FileNotFoundError:
      print("Error: output file not found.")
      exit()



