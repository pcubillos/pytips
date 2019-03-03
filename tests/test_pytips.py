import sys
import os
import pytest

import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__) + '/..') + '/'
sys.path.append(ROOT)
import pytips as t


def test_iso():
    np.testing.assert_equal(t.iso(6), np.array([211, 311, 212, 312]))


def test_pf_iso():
    molID = 6
    isoID = t.iso(molID)
    np.testing.assert_allclose(t.tips(molID, isoID, 100.0),
        np.array([116.40320395, 232.81455212, 939.16979858, 1879.84831201]),
        rtol=1e-7)


def test_pf_temp():
    molID = 6
    isoID = 211
    temps = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    np.testing.assert_allclose(t.tips(molID, isoID, temps),
        np.array([ 116.40320395, 326.64080103, 602.85201367, 954.65522363,
                  1417.76400684]), rtol=1e-7)
