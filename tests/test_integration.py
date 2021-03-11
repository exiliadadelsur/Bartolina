# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE


import bartolina

import numpy as np
import numpy.testing as npt

import pytest


# =============================================================================
# MARKERS
# =============================================================================

# ALL TESTS OF THIS MODULE ARE INTEGRATION TESTS, and SLOW as hell!
pytestmark = [pytest.mark.integration]


# ============================================================================
# TESTS
# ============================================================================


def test_integration_radius_40(full_dmh):
    dmh = full_dmh

    # halos, galingroups = bt.dark_matter_halos()
    radius = dmh[0].radius[2903]

    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_almost_equal(radius, 1.2232560617479369, 5)


def test_integration_zcenter(full_bt, full_dmh):
    bt, dmh = full_bt, full_dmh
    z = bt.z[dmh[1].groups == 2903]
    redshift_center = dmh[0].z_centers[2903]

    # center halos must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_integration_numhalo(full_dmh):
    dmh = full_dmh

    unique_elements, counts_elements = np.unique(
        dmh[1].groups, return_counts=True
    )
    numhalo = np.sum([counts_elements > 150])

    # groups with more of 150 members are 13
    assert numhalo == 13


def test_integration_halo_properties_rad(full_dmh):
    dmh = full_dmh

    # halos radius are positive
    assert np.sum(dmh[0].radius < 0) == 0


def test_integration_halo_properties_mass(full_dmh):
    dmh = full_dmh

    # mass array dimension is 1
    assert dmh[0].mass.ndim == 1

    # halos mass are positive
    assert np.sum(dmh[0].mass < 0) == 0


def test_integration_dark_matter_halos_radius(full_dmh):
    dmh = full_dmh
    assert dmh[0].radius.ndim == 1

    # radius array length
    assert len(dmh[0].radius) == 35389

    # type of value, radius array
    assert isinstance(dmh[0].radius[0], float)


def test_integration_dark_matter_halos_hmassive(full_dmh):
    dmh = full_dmh
    assert len(dmh[0].labels_h_massive[0]) == 34487


def test_integration_dc_fog_corr_len(full_table):
    table = full_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])

    halos, galingroups = rzs.dark_matter_halos()
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)

    # length of dc_fog_corr return
    assert len(dcfogcorr) == len(rzs.z)


def test_integration_z_fog_corr_len(full_table):
    table = full_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])

    halos, galingroups = rzs.dark_matter_halos()
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)

    # length of z_fog_corr return
    assert len(zfogcorr) == len(rzs.z)


def test_integration_fogcorr(full_table):
    table = full_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)

    # limits of the corrected redshift
    assert zfogcorr.min() >= 0
    assert zfogcorr.max() <= rzs.z.max()


def test_integration_fogcorr_zluminous(full_table):
    table = full_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"])
    zfogcorr = zfogcorr[table["ABSR"] < -20.5]
    z = rzs.z[table["ABSR"] < -20.5]

    # redshift of the luminous galaxies
    npt.assert_allclose(zfogcorr, z)
