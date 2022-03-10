#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3 2022

@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import pyslha # pyslha module for reading SLHA files, by Andy Buckley (https://arxiv.org/abs/1305.4194)
import pandas as pd

def read_csv(*args, **kwargs):
  """
  Function to read csv. Just a wrapper around the pandas version
  """
  return pd.read_csv(*args, **kwargs)

def read_slha(filepath):
  """
  Function to read from slha files
  """

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
               'dm31' : mNu3**2 - mNu1**2,
               'theta12'    : theta12,
               'theta23'    : theta23,
               'theta13'    : theta13,
               'delta'      : delta}
  
  return nu_params
