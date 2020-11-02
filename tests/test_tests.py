# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Perez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import numpy as np
import bartolina as bt
from astropy.table import Table
import pytest


@pytest.fixture
def obj():
    gal = Table.read("resources/SDSS.fits")
    btobj = bt.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    return btobj


def test_numHalo(obj):

    obj.Halos()
    unique_elements, counts_elements = np.unique(
        obj.clustering.labels_, return_counts=True
    )
    canthalo = np.sum([counts_elements > 150])
    assert canthalo == 15


def test_hmass(obj):

    obj.Halos()
    assert len(obj.labelshmassive[0]) == 25


def test_grid3d(obj):

    obj.Halos()
    obj.Kaisercorr()
    unique_elements, counts_elements = np.unique(
        obj.valingrid, axis=0, return_counts=True
    )
    assert len(unique_elements) == len(counts_elements)
