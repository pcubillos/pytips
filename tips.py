import sys
import numpy as np
from numpy.core.numeric import isscalar

sys.path.append("lib/")
import ctips as ct


def tips(molID, isoID, temp):
  """
  Evaluate the partition function for the given isotope(s) at the given
  temperature(s).  This is a wrapper of ctips.tips.

  Parameters:
  -----------
  molID: Scalar or iterable
     The molecular ID as given by HITRAN 2012.
  isoID: Scalar or iterable
     The isotope ID (AFGL) as given by HITRAN 2012.
  temp:  Scalar or iterable
     Temperature a which to evaluate the partition function.

  Notes:
  ------
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


def iso(molID):
  """
  Return the list of isotope IDs for the given molecule ID.
  Wrapper of C ctips iso function.
  """
  return ct.iso(int(molID))
