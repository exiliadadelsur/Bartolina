# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Perez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import numpy as np
import bartolina
from astropy.table import Table
import pytest


@pytest.fixture
def bt():
    gal = Table.read("resources/SDSS.fits")
    rzs = bartolina.re_z_space(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    return rzs


def test_numHalo(bt):

    bt.aalos()
    unique_elements, counts_elements = np.unique(
        bt.clustering.labels_, return_counts=True
    )
    canthalo = np.sum([counts_elements > 150])
    assert canthalo == 15


def test_hmass(bt):

    bt.aalos()
    assert len(bt.labelshmassive[0]) == 25


def test_grid3d(bt):

    bt.aalos()
    bt.kaisercorr()
    unique_elements, counts_elements = np.unique(
        bt.valingrid, axis=0, return_counts=True
    )
    assert len(unique_elements) == len(counts_elements)


def test_FoGcorr(bt):

    bt.halos()
    dcCorr, zCorr = bt.fogcorr(seedvalue=1234)
    assert dcCorr.max() < 1120
