# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import os
import pathlib

from astropy.table import Table

import bartolina

import numpy as np
from numpy.testing import assert_almost_equal

import pytest


# ==============================================================================
# CONSTANTS
# ==============================================================================


PATH = os.path.abspath(os.path.dirname(__file__))

RESOURCES_PATH = pathlib.Path(PATH).parents[0] / "resources"


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def bt():
    gal = Table.read(RESOURCES_PATH / "SDSS.fits")
    rzs = bartolina.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    return rzs


# ==============================================================================
# TESTS
# ==============================================================================


def test_radius(bt):
    xyz = bt._xyzcoordinates()
    groups = bt._groups(xyz)
    # identify id cluster with 40 members
    idcl, numcl = np.unique(groups, return_counts=True)
    idcl = idcl[numcl <= 40]
    numcl = numcl[numcl <= 40]
    idcl[np.argsort(numcl)[::-1][0]]
    idclmax = idcl[np.argsort(numcl)[::-1][0]]
    # radius calculation
    radius = bt._radius(
        bt.ra[groups == idclmax],
        bt.dec[groups == idclmax],
        bt.z[groups == idclmax],
    )
    # halo with 40 members must have a virial radius less than 3 Mpc
    assert radius < 3


def test_zcenter(bt):
    xyz = bt._xyzcoordinates()
    groups = bt._groups(xyz)
    # identify id cluster with 40 members
    idcl, numcl = np.unique(groups, return_counts=True)
    idcl = idcl[numcl <= 40]
    numcl = numcl[numcl <= 40]
    idcl[np.argsort(numcl)[::-1][0]]
    idclmax = idcl[np.argsort(numcl)[::-1][0]]
    # find centers and their redshifts
    xyz = xyz[groups == idclmax]
    z = bt.z[groups == idclmax]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, z)
    # center must have redshift between halo's redshifts limits
    assert redshift_center < z.max()


def test_numhalo(bt):
    xyz = bt._xyzcoordinates()
    groups = bt._groups(xyz)
    unique_elements, counts_elements = np.unique(groups, return_counts=True)
    numhalo = np.sum([counts_elements > 150])
    assert numhalo == 14


def test_hmass(bt):
    xyz = bt._xyzcoordinates()
    groups = bt._groups(xyz)
    # identify id cluster with 40 members
    idcl, numcl = np.unique(groups, return_counts=True)
    idcl = idcl[numcl <= 40]
    numcl = numcl[numcl <= 40]
    idcl[np.argsort(numcl)[::-1][0]]
    idclmax = idcl[np.argsort(numcl)[::-1][0]]

    radius = bt._radius(
        bt.ra[groups == idclmax],
        bt.dec[groups == idclmax],
        bt.z[groups == idclmax],
    )

    xyz = xyz[groups == idclmax]
    z = bt.z[groups == idclmax]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, z)

    hmass = bt._halomass(radius, redshift_center)

    assert hmass < 10 ** 16


def test_bias(bt):

    bias = bt._bias(100, 10 ** 12.5, 0.27)
    expected_bias = np.array([1.00714324])  # 8 decimals

    assert_almost_equal(bias, expected_bias, 8)  # equal within 8 decimals


# def test_grid3d(bt):

#    bt.halos()
#    bt.kaisercorr()
#    unique_elements, counts_elements = np.unique(
#        bt.valingrid, axis=0, return_counts=True
#    )
#    assert len(unique_elements) == len(counts_elements)


# def test_FoGcorr(bt):

#    bt.halos()
#    dcCorr, zCorr = bt.fogcorr(seedvalue=1234)
#    assert dcCorr.max() < 1120
