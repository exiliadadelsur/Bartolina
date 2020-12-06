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
    xyz = bt.xyzcoordinates()
    groups, id_groups = bt.groups(xyz)
    radius = bt.radius(
        bt.ra[groups == 2903],
        bt.dec[groups == 2903],
        bt.z[groups == 2903],
    )
    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_almost_equal(radius, 1.2232560617479369, 5)


def test_hmass_40(bt):
    xyz = bt.xyzcoordinates()
    groups, id_groups = bt.groups(xyz)
    radius = bt.radius(
        bt.ra[groups == 2903],
        bt.dec[groups == 2903],
        bt.z[groups == 2903],
    )
    xyz = xyz[groups == 2903]
    z = bt.z[groups == 2903]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt.centers(xyz, z)
    hmass = bt.halomass(radius, redshift_center)
    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_approx_equal(hmass / 10 ** 14, 1, significant=0.1)


def test_zcenter(bt):
    xyz = bt.xyzcoordinates()
    groups, id_groups = bt.groups(xyz)
    xyz = xyz[groups == 2903]
    z = bt.z[groups == 2903]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt.centers(xyz, z)
    # center halos must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_numhalo(bt):
    xyz = bt.xyzcoordinates()
    groups, id_groups = bt.groups(xyz)
    unique_elements, counts_elements = np.unique(groups, return_counts=True)
    numhalo = np.sum([counts_elements > 150])
    # groups with more of 150 members are 13
    assert numhalo == 13


@pytest.mark.haloprop
def test_halo_properties_rad(bt):
    xyz = bt.xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt.groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt.group_prop(id_groups, groups, xyz)
    # halos radius are positive
    assert np.sum(rad < 0) == 0


@pytest.mark.haloprop
def test_halo_properties_mass(bt):
    xyz = bt.xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt.groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt.group_prop(id_groups, groups, xyz)
    # mass array dimension is 1
    assert mass.ndim == 1
    # halos mass are positive
    assert np.sum(mass < 0) == 0


@pytest.mark.haloprop
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


@pytest.mark.thisis
def test_bias(bt):
    bias = bt.bias(100, 10 ** 12.5, 0.27)
    expected_bias = np.array([1.00714324])  # 8 decimals
    npt.assert_almost_equal(bias, expected_bias, 8)  # equal within 8 decimals


def test_dc_fog_corr_len(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    halos, galingroups = rzs._dark_matter_halos()
    dcfogcorr, dc_centers, radius, groups = rzs._dc_fog_corr(
        table["ABSR"], halos, galingroups, halos.dc_centers, seedvalue=26
    )
    # length of _dc_fog_corr return
    assert len(dcfogcorr) == len(rzs.z)


def test_z_fog_corr_len(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    halos, galingroups = rzs._dark_matter_halos()
    dcfogcorr, dc_centers, radius, groups = rzs._dc_fog_corr(
        table["ABSR"], halos, galingroups, halos.dc_centers, seedvalue=26
    )
    zfogcorr = rzs._z_fog_corr(
        dcfogcorr,
        table["ABSR"],
        halos,
        galingroups,
    )
    # length of _z_fog_corr return
    assert len(zfogcorr) == len(rzs.z)


def test_fogcorr(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)
    # limits of the corrected redshift
    assert zfogcorr.min() >= 0
    assert zfogcorr.max() <= rzs.z.max()


def test_fogcorr_zluminous(table):
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"])
    zfogcorr = zfogcorr[table["ABSR"] < -20.5]
    z = rzs.z[table["ABSR"] < -20.5]
    # redshift of the luminous galaxies
    npt.assert_allclose(zfogcorr, z)


def test_density(bt):
    valingrid = np.array([[1, 2, 3], [4, 3, 1], [0, 3, 4]])
    hmass = np.array([12, 50, 15])
    delta = bt.density(valingrid, hmass, 5)
    array = np.array(
        [
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            50.0,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            12.0,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            15.0,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
            0.616,
        ]
    )
    npt.assert_almost_equal(delta, array)


def test_f(bt):
    f = bt.calcf(0.27, 0.73)
    npt.assert_almost_equal(f, 0.4690904014151921, 10)


def test_zkaisercorr(bt):
    z = np.array([0.1, 0.12, 0.09])
    v = np.array([1.58, 1.7, 1.63])
    array = np.array([0.09999999, 0.11999999, 0.08999999])
    zcorr = bt.zkaisercorr(z, v)
    npt.assert_almost_equal(array, zcorr)


@pytest.mark.webtest
def test_grid3daxislim(bt):
    halos, galingroups = bt.dark_matter_halos()
    centers, labels = halos.xyzcenters, halos.labels_h_massive
    inf, sup = bt.grid3d_axislim(centers, labels)
    i = np.array([-564.0966701, -515.7190963, -32.9904224])
    s = np.array([-13.50295, 479.20005, 522.83549])
    npt.assert_almost_equal(inf, i, decimal=5)
    npt.assert_almost_equal(sup, s, decimal=5)


# @pytest.mark.webtest
def test_grid3dgridlim(bt):
    halos, galingroups = bt.dark_matter_halos()
    centers, labels = halos.xyzcenters, halos.labels_h_massive
    inf, sup = bt.grid3d_axislim(centers, labels)
    liminf, limsup = bt.grid3d_gridlim(inf, sup)
    i = np.array([-836.2593814, -565.7190963, -302.5370377])
    s = np.array([258.659766, 529.2000511, 792.3821097])
    npt.assert_almost_equal(liminf, i)
    npt.assert_almost_equal(limsup, s)


# @pytest.mark.webtest
def test_grid3dcells(bt):
    liminf = np.array([0, 0, 0])
    limsup = np.array([5, 5, 5])
    nbines = 6
    centers = np.array(
        [[1.1, 0.1, 2.4], [3.5, 4.6, 3.2], [2.1, 3.7, 1.1], [1, 2, 1]]
    )
    array = np.array([[[1, 0, 2]], [[4, 5, 3]], [[2, 4, 1]], [[1, 2, 1]]])
    valingrid = bt.grid3dcells(liminf, limsup, centers, nbines)
    npt.assert_almost_equal(valingrid, array)


# @pytest.mark.webtest
def test_grid3d(bt):
    centers = np.array(
        [[1.1, 0.1, 2.4], [3.5, 4.6, 3.2], [2.1, 3.7, 1.1], [1, 2, 1]]
    )
    labels = np.array([0, 1, 2, 3])
    array = np.array(
        [
            [[500, 489, 514]],
            [[524, 534, 522]],
            [[510, 525, 502]],
            [[499, 508, 501]],
        ]
    )
    valingrid = bt.grid3d(centers, labels)
    npt.assert_almost_equal(valingrid, array)
