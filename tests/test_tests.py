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
    obj = bartolina.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    return obj


def test_numHalo(bt):

    bt.Halos()
    unique_elements, counts_elements = np.unique(
        bt.clustering.labels_, return_counts=True
    )
    canthalo = np.sum([counts_elements > 150])
    assert canthalo == 15


def test_hmass(bt):

    bt.Halos()
    assert len(bt.labelshmassive[0]) == 25


def test_grid3d(bt):

    bt.Halos()
    bt.Kaisercorr()
    unique_elements, counts_elements = np.unique(
        bt.valingrid, axis=0, return_counts=True
    )
    assert len(unique_elements) == len(counts_elements)
    
def test_FoGcorr():

    gal = Table.read("resources/SDSS.fits")
    obj = bt.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    obj.Halos()    
    dcCorr, zCorr = obj.FoGcorr()      
    assert ((dcCorr.min()>25) * (dcCorr.max()<890))
    
  
    
#@pytest.mark.xfail
#@pytest.mark.xfail(not FLAG, reason=
#try:
#    import xyz
#except:
#    FLAG=False
