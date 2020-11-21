# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

from astropy.table import Table

import bartolina

import numpy as np
import numpy.testing as npt

import pytest


@pytest.fixture(scope="session")
def table():
    gal = Table.read("../resources/SDSS.fits")
    return gal

@pytest.fixture(scope="session")
def bt(table):
    r = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    return r

#@pytest.fixture(scope="session")
#def bt():
#    gal = Table.read("../resources/SDSS.fits")
#    rzs = bartolina.ReZSpace(gal["RAJ2000"], gal["DEJ2000"], gal["zobs"])
#    mags = gal['ABSR']
#    return rzs, mags


def test_radius_40(bt):
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    radius = bt._radius(
        bt.ra[groups == 2903],
        bt.dec[groups == 2903],
        bt.z[groups == 2903],
    )
    # halo with 40 members must have a virial radius less than 3 Mpc
    assert radius == 0
#    assert radius > 0


def test_hmass_40(bt):   
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    radius = bt._radius(
        bt.ra[groups == 2903],
        bt.dec[groups == 2903],
        bt.z[groups == 2903],
    )

    xyz = xyz[groups == 2903]
    z = bt.z[groups == 2903]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, z)

    hmass = bt._halomass(radius, redshift_center)

    #assert hmass < 0
    npt.assert_approx_equal(hmass/10 ** 14, 1, significant= 0.1)


def test_zcenter(bt):   
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    xyz = xyz[groups == 2903]
    z = bt.z[groups == 2903]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, z)
    # center must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_numhalo(bt):   
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    unique_elements, counts_elements = np.unique(groups, return_counts=True)
    numhalo = np.sum([counts_elements > 150])
    assert numhalo == 13
 
    
def test_halo_properties_rad(bt):   
    xyz = bt._xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt._groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt._group_prop(id_groups, groups, xyz)    
    assert np.sum(rad < 0) == 0

def test_halo_properties_mass(bt):   
    xyz = bt._xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt._groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt._group_prop(id_groups, groups, xyz)    
    assert np.sum(mass < 0) == 0
    
# def test_bias(bt):

#    bias = bt._bias(100, 10 ** 12.5, 0.27)
#    assert bias == np.array([1.00714324])

# def test_grid3d(bt):

#    bt.halos()
#    bt.kaisercorr()
#    unique_elements, counts_elements = np.unique(
#        bt.valingrid, axis=0, return_counts=True
#    )
#    assert len(unique_elements) == len(counts_elements)

def test_dc_fog_corr(table, bt):
    dcfogcorr, dc_centers, radius, groups = bt._dc_fog_corr(table['ABSR'])    
    delta = np.abs(dc_centers[2903] - radius[2903])
    mask = (groups == 2903)
    galradius = np.abs(dc_centers[2903] - dcfogcorr[mask])
    assert galradius.max() <= delta 
    assert len(dcfogcorr) == len(bt.z)


#def test_z_fog_corr(table, bt):
#    dcfogcorr, dc_centers, radius, groups = bt._dc_fog_corr(table['ABSR'])
#    zfogcorr = bt._z_fog_corr(dcfogcorr, table['ABSR'])    
#    assert len(zfogcorr) == len(bt.z)


#def test_fogcorr(table, bt):
#    dcfogcorr, zfogcorr = bt.fogcorr(table['ABSR'])
#    assert zfogcorr.min() > 0
#    assert zfogcorr.max() < bt.z.max()






