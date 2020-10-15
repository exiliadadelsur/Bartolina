import numpy as np
import bartolina as bt
from astropy.table import Table


def test_numHalo():

    gal = Table.read("resources/SDSS.fits")
    obj = bt.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["z"],
                      100.0, 0.27, 0.73)
    obj.mclustering()
    unique_elements, counts_elements = np.unique(
        obj.clustering.labels_, return_counts=True
    )
    canthalo = np.sum([counts_elements > 150])
    assert canthalo == 15
