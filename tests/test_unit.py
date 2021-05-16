# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

import bartolina

import numpy as np
import numpy.testing as npt


# ============================================================================
# TESTS
# ============================================================================


def test_fof():
    random = np.random.default_rng(seed=42)
    npts = 10

    cluster0 = random.normal(-1, 0.2, (npts // 2, 2))
    cluster1 = random.normal(1, 0.2, (npts // 2, 2))
    data = np.vstack((cluster0, cluster1))

    fof = bartolina.FoF(0.4)
    labels = fof.fit_predict(data)

    npt.assert_array_equal(labels, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def test_radius_40(sample_dmh):
    dmh = sample_dmh
    radius = dmh[0].radius[150]

    # halo with 40 members must have a virial radius less than 3 Mpc
    npt.assert_almost_equal(radius, 2.355721913769786)


def test_zcenter(sample_bt, sample_dmh):
    bt, dmh = sample_bt, sample_dmh
    z = bt.z[dmh[1].groups == 150]
    redshift_center = dmh[0].z_centers[150]

    # center halos must have redshift between halo's redshifts limits
    assert redshift_center < z.max()
    assert redshift_center > z.min()


def test_numhalo(sample_dmh):
    dmh = sample_dmh
    unique_elements, counts_elements = np.unique(
        dmh[1].groups, return_counts=True
    )
    numhalo = np.sum([counts_elements > 150])

    # groups with more of 150 members are 1
    assert numhalo == 1


def test_halo_properties_rad(sample_dmh):
    dmh = sample_dmh

    # halos radius are positive
    assert np.sum(dmh[0].radius < 0) == 0


def test_halo_properties_mass(sample_dmh):
    dmh = sample_dmh

    # mass array dimension is 1
    assert dmh[0].mass.ndim == 1

    # halos mass are positive
    assert np.sum(dmh[0].mass < 0) == 0


def test_dark_matter_halos_radius(sample_dmh):
    dmh = sample_dmh
    assert dmh[0].radius.ndim == 1

    # radius array length
    assert len(dmh[0].radius) == 359

    # type of value, radius array
    assert isinstance(dmh[0].radius[0], float)


def test_dark_matter_halos_hmassive(sample_dmh):
    dmh = sample_dmh
    assert len(dmh[0].labels_h_massive[0]) == 352


def test_fogcorr(sample_table):
    table = sample_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])

    dcfogcorr, zfogcorr = rzs.fogcorr(table["ABSR"], seedvalue=26)

    # length of dc_fog_corr return
    assert len(dcfogcorr) == len(rzs.z)

    # length of z_fog_corr return)
    assert len(zfogcorr) == len(rzs.z)

    # limits of the corrected redshift
    assert zfogcorr.min() >= 0
    assert zfogcorr.max() <= rzs.z.max()

    # redshift of the luminous galaxies
    zfogcorr = zfogcorr[table["ABSR"] < -20.5]
    z = rzs.z[table["ABSR"] < -20.5]

    npt.assert_allclose(zfogcorr, z)


def test_kaisercorr(sample_table):
    table = sample_table
    table = table[table["ABSR"] > -20.6]
    table = table[table["ABSR"] < -20.4]
    rzs = bartolina.ReZSpace(table["RAJ2000"], table["DEJ2000"], table["zobs"])
    dckaisercorr, zkaisercorr = rzs.kaisercorr(n_cells=3)
    halos, galinhalo = rzs.dark_matter_halos()

    # length of dckaisercorr return
    assert len(dckaisercorr) == len(halos.z_centers)

    # length of zkaisercorr return
    assert len(zkaisercorr) == len(halos.z_centers)

    # limits of the corrected redshift
    assert zkaisercorr.min() >= 0
