# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Perez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import numpy as np
import bartolina as bt
from astropy.table import Table


def test_numHalo():

    gal = Table.read("resources/SDSS.fits")
    obj = bt.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    obj.mclustering()
    unique_elements, counts_elements = np.unique(
        obj.clustering.labels_, return_counts=True
    )
    canthalo = np.sum([counts_elements > 150])
    assert canthalo == 15


def test_hmass():

    gal = Table.read("resources/SDSS.fits")
    obj = bt.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"])
    obj.mclustering()
    assert len(obj.hmass) == 24
