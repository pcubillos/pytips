# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# pytips is open-source software under the MIT license (see LICENSE).

__all__ = ["tips", "iso", "molID", "molname", "to_file"]

import sys, os
import numpy as np
from numpy.core.numeric import isscalar

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/lib")
import ctips as ct

_molname = np.array(["", "H2O",    "CO2",    "O3",    "N2O",    "CO",
                         "CH4",    "O2",     "NO",    "SO2",    "NO2",
                         "NH3",    "HNO3",   "OH",    "HF",     "HCl",
                         "HBr",    "HI",     "ClO",   "OCS",    "H2CO",
                         "HOCl",   "N2",     "HCN",   "CH3Cl",  "H2O2",
                         "C2H2",   "C2H6",   "PH3",   "COF2",   "SF6",
                         "H2S",    "HCOOH",  "HO2",   "O",      "ClONO2",
                         "NO+",    "HOBr",   "C2H4",  "CH3OH",  "CH3Br",
                         "CH3CN",  "CF4",    "C4H2",  "HC3N",   "H2",
                         "CS",     "SO3",    "C2N2",  "SO",     "C3H4",
                         "CH3",    "CS2"])

def tips(molID, isoID, temp):
  """
  Evaluate the partition function for the given isotope(s) at the given
  temperature(s).  This is a wrapper of ctips.tips.

  Parameters
  ----------
  molID: Scalar or iterable
     The molecular ID as given by HITRAN 2012.
  isoID: Scalar or iterable
     The isotope ID (AFGL) as given by HITRAN 2012.
  temp:  Scalar or iterable
     Temperature a which to evaluate the partition function.

  Notes
  -----
  - The molID and isoID are casted into an integer ndarray data types.
  - The temp is casted into a double ndarray data type.
  - If the arguments have different sizes, the code resizes them to
    a same size, unless they have incompatible sizes.
  """
  # Check scalar vs iterable, turn into iterable:
  if isscalar(molID):
    molID = [molID]
  if isscalar(isoID):
    isoID = [isoID]
  if isscalar(temp):
    temp = [temp]

  # Turn them numpy arrays:
  molID = np.asarray(molID, np.int)
  isoID = np.asarray(isoID, np.int)
  temp  = np.asarray(temp,  np.double)

  # Set them to the same size:
  if len(isoID) != len(temp):
    if   len(isoID) == 1:
      isoID = np.repeat(isoID, len(temp))
    elif len(temp)  == 1:
      temp  = np.repeat(temp,  len(isoID))
    else:
      sys.exit(0)

  if len(molID) != len(isoID):
    if len(molID) != 1:
      sys.exit(0)
    molID = np.repeat(molID, len(isoID))

  return ct.tips(molID, isoID, temp)


def iso(mID):
  """
  Get the list of isotope IDs for the given molecule ID.

  Parameters
  ----------
  mID: String or integer
    Molecule name (if string) or molecule ID.

  Return
  ------
  isoID: 1D integer ndarray
    Isotopes ID for molecule mID.
  """
  if isinstance(mID, str):
    # Convert string to index if necesssary:
    return ct.iso(int(molID(mID)))
  return ct.iso(int(mID))


def molID(mname):
  """
  Get the molecule ID for the requested molecule.

  Parameters
  ----------
  mname: String
    Name of the molecule.

  Return
  ------
  mID: Integer
    The molecule's ID.
  """
  if mname not in _molname:
    print("Molecule '{:s}' is not in list.".format(mname))
    return None
  return np.where(_molname == mname)[0][0]


def molname(mID):
  """
  Get the molecule name for the requested molecule ID.

  Parameters
  ----------
  mID: Integer
    The molecule's ID.

  Return
  ------
  mname: String
    Name of the molecule.
  """
  if (mID < 1) or (mID > 52):
    print("Molecule ID '{:d}' is invalid.".format(mID))
    return None
  return _molname[mID]


def to_file(filename, molname, temp):
  """
  Compute partition-function values for all isotopes of a given
  molecule over a temperature array, and save to file.

  Parameters
  ----------
  filename: String
     Output partition-function file.
  molname: String
     Name of the molecule.
  temp: 1D float ndarray
     Array of temperatures.

  Example
  -------
  >>> import pytips as p
  >>> temp = np.linspace(70, 3000, 294)
  >>> molname = "CO2"
  >>> p.to_file("CO2_tips.dat", molname, temp)
  """
  # Compute partition function:
  isoID = iso(molname)
  niso  = len(isoID)
  ntemp = len(temp)
  data = np.zeros((niso, ntemp), np.double)
  for i in np.arange(niso):
    data[i] = tips(molID(molname), isoID[i], temp)

  # Save to file:
  with open(filename, "w") as fout:
    fout.write(
    "# Tabulated {:s} partition-function data from TIPS.\n\n".format(molname))

    fout.write("@ISOTOPES\n         ")
    for j in np.arange(niso):
        fout.write("  {:10s}".format(str(isoID[j])))
    fout.write("\n\n")

    fout.write("# Temperature (K), partition function for each isotope:\n")
    fout.write("@DATA\n")
    for i in np.arange(ntemp):
      fout.write(" {:7.1f} ".format(temp[i]))
      for j in np.arange(niso):
        fout.write("  {:10.4e}".format(data[j,i]))
      fout.write("\n")
