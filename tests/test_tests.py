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
@pytest.mark.porfavorfunciona
def test_pmesh(bt):
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, bt.z)
    import pmesh
    # Creo la grilla
    pm = pmesh.ParticleMesh(BoxSize=500,Nmesh=np.array([20,20,20]))
    smoothing = 1.0 * pm.Nmesh / pm.BoxSize
    # Ubico los puntos en la grilla
    layout = pm.decompose(xyz)
    # Transformación real a compleja
    density = pm.create(type='real')
    density.paint(xyz, layout=layout)
    # Se obtiene un campo complejo (será la trasnformada de fourier)
    def potential_transfer_function(k, v):
        k2 = k.normp(zeromode=1)
        return v / (k2)

    pot_k = density.r2c(out=Ellipsis).apply(potential_transfer_function, out=Ellipsis)
    # Se obtiene la velocidad en el espacio real (supongo)
    for d in range(3):
        def force_transfer_function(k, v, d=d):
            return k[d] * 1j * v

        force_d = pot_k.apply(force_transfer_function).c2r(out=Ellipsis)

        w = force_d.readout(xyz, layout=layout)
    npt.assert_allclose(w, bt.z)

def test_radius_40(bt):
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    radius = bt._radius(
        bt.ra[groups == 2903],
        bt.dec[groups == 2903],
        bt.z[groups == 2903],
    )
    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_almost_equal(radius, 1.2232560617479369, 5)


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
    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_approx_equal(hmass / 10 ** 14, 1, significant=0.1)


def test_zcenter(bt):
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    xyz = xyz[groups == 2903]
    z = bt.z[groups == 2903]
    xcen, ycen, zcen, dc_center_i, redshift_center = bt._centers(xyz, z)
    # center halos must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_numhalo(bt):
    xyz = bt._xyzcoordinates()
    groups, id_groups = bt._groups(xyz)
    unique_elements, counts_elements = np.unique(groups, return_counts=True)
    numhalo = np.sum([counts_elements > 150])
    # groups with more of 150 members are 13
    assert numhalo == 13


def test_halo_properties_rad(bt):
    xyz = bt._xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt._groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt._group_prop(id_groups, groups, xyz)
    # halos radius are positive
    assert np.sum(rad < 0) == 0


def test_halo_properties_mass(bt):
    xyz = bt._xyzcoordinates()
    # finding group of galaxies
    groups, id_groups = bt._groups(xyz)
    # mass and center, for each group
    xyz_c, dc_c, z_c, rad, mass = bt._group_prop(id_groups, groups, xyz)
    # mass array dimension is 1
    assert mass.ndim == 1
    # halos mass are positive
    assert np.sum(mass < 0) == 0


def test_dark_matter_halos_radius(bt):
    halos, galingroups = bt._dark_matter_halos()
    # radius array dimension is 1
    assert halos.radius.ndim == 1
    # radius array length
    assert len(halos.radius) == 35389
    # type of value, radius array
    assert isinstance(halos.radius[0], float)


def test_dark_matter_halos_hmassive(bt):
    halos, galingroups = bt._dark_matter_halos()
    # massive halos array length
    assert len(halos.labels_h_massive[0]) == 34487


def test_bias(bt):
    bias = bt._bias(100, 10 ** 12.5, 0.27)
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


# def test_realspace_lendc(table):
#    table = table[table["ABSR"] > -20.6]
#    table = table[table["ABSR"] < -20.4]
#    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"],
#    table["zobs"])
#    dc, zcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)
# length of the corrected comoving distance array
#    assert len(dc) == len(rzs.z)


# def test_realspace_lenzcorr(table):
#    table = table[table["ABSR"] > -20.6]
#    table = table[table["ABSR"] < -20.4]
#    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"],
#    table["zobs"])
#    halos, galingroups = rzs._dark_matter_halos()
#    dc, zcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)
#    z = rzs._z_realspace(dc, halos, galingroups)
# length of the corrected redshift array
#    assert len(z) == len(rzs.z)


def test_density(bt):
    valingrid = np.array([[1, 2, 3], [4, 3, 1], [0, 3, 4]])
    hmass = np.array([12, 50, 15])
    delta = bt._density(valingrid, hmass, 5)
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
    f = bt._calcf(0.27, 0.73)
    npt.assert_almost_equal(f, 0.4690904014151921, 10)


def test_zkaisercorr(bt):
    z = np.array([0.1, 0.12, 0.09])
    v = np.array([1.58, 1.7, 1.63])
    array = np.array([0.09999999, 0.11999999, 0.08999999])
    zcorr = bt._zkaisercorr(z, v)
    npt.assert_almomost_equal(array, zcorr)


@pytest.mark.webtest
def test_grid3daxislim(bt):
    halos, galingroups = bt._dark_matter_halos()
    centers, labels = halos.xyzcenters, halos.labels_h_massive
    inf, sup = bt._grid3d(centers, labels)
    i = 1
    s = 3
    npt.assert_almost_equal(inf, i)
    npt.assert_almost_equal(sup, s)
