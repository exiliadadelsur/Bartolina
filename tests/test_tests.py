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
import numpy.testing as npt

import pytest


# ============================================================================
# CONSTANTS
# ============================================================================


PATH = os.path.abspath(os.path.dirname(__file__))

RESOURCES_PATH = pathlib.Path(PATH).parents[0] / "resources"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def table():
    gal = Table.read(RESOURCES_PATH / "SDSS.fits")
    return gal


@pytest.fixture(scope="session")
def bt(table):
    r = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    return r


# ============================================================================
# TESTS
# ============================================================================


def test_radius_40(bt):
    halos, galingroups = bt.dark_matter_halos()
    radius = halos.radius[2903]
    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_almost_equal(radius, 1.2232560617479369, 5)


def test_zcenter(bt):
    halos, galingroups = bt.dark_matter_halos()
    z = bt.z[galingroups.groups == 2903]
    redshift_center = halos.z_centers[2903]
    # center halos must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_numhalo(bt):
    halos, galingroups = bt.dark_matter_halos()   
    unique_elements, counts_elements = np.unique(galingroups.groups, 
                                                 return_counts=True)
    numhalo = np.sum([counts_elements > 150])
    # groups with more of 150 members are 13
    assert numhalo == 13


def test_halo_properties_rad(bt):
    halos, galingroups = bt.dark_matter_halos() 
    # halos radius are positive
    assert np.sum(halos.radius < 0) == 0



def test_halo_properties_mass(bt):
    halos, galingroups = bt.dark_matter_halos() 
    # mass array dimension is 1
    assert halos.mass.ndim == 1
    # halos mass are positive
    assert np.sum(halos.mass < 0) == 0



def test_dark_matter_halos_radius(bt):
    halos, galingroups = bt.dark_matter_halos()
    # radius array dimension is 1
    assert halos.radius.ndim == 1
    # radius array length
    assert len(halos.radius) == 35389
    # type of value, radius array
    assert isinstance(halos.radius[0], float)


@pytest.mark.haloprop
def test_dark_matter_halos_hmassive(bt):
    halos, galingroups = bt.dark_matter_halos()
    # massive halos array length
    assert len(halos.labels_h_massive[0]) == 34487


def test_dc_fog_corr_len(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    halos, galingroups = rzs.dark_matter_halos()
    dcfogcorr, zfogcorr= rzs.fogcorr(
        table["ABSR"], seedvalue=26
    )
    # length of dc_fog_corr return
    assert len(dcfogcorr) == len(rzs.z)


@pytest.mark.thisis
def test_z_fog_corr_len(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    halos, galingroups = rzs.dark_matter_halos()
    dcfogcorr, zfogcorr= rzs.fogcorr(
        table["ABSR"], seedvalue=26
    )
    # length of z_fog_corr return
    assert len(zfogcorr) == len(rzs.z)


#@pytest.mark.thisis
def test_fogcorr(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)
    # limits of the corrected redshift
    assert zfogcorr.min() >= 0
    assert zfogcorr.max() <= rzs.z.max()


#@pytest.mark.thisis
def test_fogcorr_zluminous(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"])
    zfogcorr = zfogcorr[table["ABSR"] < -20.5]
    z = rzs.z[table["ABSR"] < -20.5]
    # redshift of the luminous galaxies
    npt.assert_allclose(zfogcorr, z)


#@pytest.mark.thisis
#def test_zkaisercorr(bt):
#    z = np.array([0.1, 0.12, 0.09])
#    v = np.array([1.58, 1.2, 1.7])
#    array = np.array([0.09999999, 0.11999999, 0.08999999])
#    zcorr = bt.zkaisercorr(z, v)
#    npt.assert_almost_equal(array, zcorr)


#@pytest.mark.thisis
#def test_kaisercorr(table):
#    table = table[table["ABSR"] > -20.6]
#    table = table[table["ABSR"] < -20.4]
#    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
#    halos, galingroups = rzs.dark_matter_halos()
#    dc, zcorr = rzs.kaisercorr()
#    assert len(dc) == len(halos.dc_centers)


#@pytest.mark.thisis
#def test_realspace(table):
#    table = table[table["ABSR"] > -20.6]
#    table = table[table["ABSR"] < -20.4]
#    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
#    dc, zcorr = rzs.realspace(table["ABSR"])
#    assert len(zcorr) == len(rzs.z)
