# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Rocío Perez and Claudio Antonio Lopez Cortez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

"""Bartolina : real space reconstruction algorithm for redshift."""

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM, z_at_value

import attr

import camb
from camb import model

from cluster_toolkit import bias

from halotools.empirical_models import NFWProfile

import numpy as np

import pandas as pd

from sklearn.cluster import DBSCAN


@attr.s(frozen=True)
class Halo(object):
    """Properties of dark matter halos"""

    xyzcenters = attr.ib()
    dc_centers = attr.ib()
    z_centers = attr.ib()
    radius = attr.ib()
    mass = attr.ib()
    labels_h_massive = attr.ib()


@attr.s(frozen=True)
class GalInGroups(object):
    """Clustering results"""

    groups = attr.ib()
    id_groups = attr.ib()


# ============================================================================
# MAIN CLASS
# ============================================================================


@attr.s
class ReZSpace(object):
    """Real space reconstruction algorithm.

    ...

    Attributes
    ----------
    ra : array_like
         Right ascension astronomy coordinate in decimal degrees.
    dec : array_like
          Declination astronomy coordinate in decimal degrees.
    z : array_like
        Observational redshift.
    cosmo : object, optional
            Instance of an astropy cosmology. Default cosmology is
            LambdaCDM with H0=100, Om0=0.27, Ode0=0.73.
    Mth : float, optional
          The threshold mass that determines massive halos in solar mass.
          Default is 10 ** 12.5.
    delta_c : string, optional
              Overdensity constant. Default is "200m".

    Methods
    -------
    kaisercorr()
        Corrects the Kaiser effect only.
    fogcorr()
        Corrects the Finger of God effect only.
    realspace()
        Corrects both effects (Kaiser and FoG).

    """

    # User input params
    ra = attr.ib()
    dec = attr.ib()
    z = attr.ib()
    cosmo = attr.ib(default=LambdaCDM(H0=100, Om0=0.27, Ode0=0.73))
    Mth = attr.ib(default=(10 ** 12.5))
    delta_c = attr.ib(default="200m")

    def __attrs_post_init__(self):
        """Find properties of massive dark matter halos.

        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.

        """
        # cartesian coordinates for galaxies
        xyz = self._xyzcoordinates()
        # finding group of galaxies
        groups, id_groups = self._groups(xyz)
        # mass and center, for each group
        xyz_c, dc_c, z_c, rad, mass = self._group_prop(id_groups, groups, xyz)
        # selec massive halos
        labels_h_massive = np.where(mass > self.Mth)
        GalInGroups(groups, id_groups)
        Halo(xyz_c, dc_c, z_c, rad, mass, labels_h_massive)

    # ========================================================================
    # Internal methods
    # ========================================================================

    def _xyzcoordinates(self):

        dc = self.cosmo.comoving_distance(self.z)
        c = SkyCoord(
            ra=np.array(self.ra) * u.degree,
            dec=np.array(self.dec) * u.degree,
            distance=np.array(dc) * u.mpc,
        )
        xyz = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
        return xyz

    def _groups(self, xyz):

        pesos = self.z * 100
        clustering = DBSCAN(eps=1.2, min_samples=24)
        clustering.fit(xyz, sample_weight=pesos)
        unique_elements, counts_elements = np.unique(
            clustering.labels_, return_counts=True
        )
        unique_elements = unique_elements[unique_elements > -1]
        return clustering.labels_, unique_elements

    def _radius(self, ra, dec, z):

        galnum = len(ra)
        dc = self.cosmo.comoving_distance(z)

        c1 = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)
        c2 = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)
        sum_rij = 0
        indi = np.arange(galnum)
        for i in indi:
            sep = c1[i].separation(c2[np.where(indi > i)])
            rp_rad = sep.radian
            rp_mpc = dc[i] * rp_rad
            rij = 1 / rp_mpc.value
            sum_rij = sum_rij + np.sum(rij)

        radius = (galnum * (galnum - 1) / (sum_rij)) * (np.pi / 2)
        return radius

    def _centers(self, xyz, z):
        xcenter = np.mean(xyz[:, 0])
        ycenter = np.mean(xyz[:, 1])
        zcenter = np.mean(xyz[:, 2])
        dc_center_i = np.sqrt(xcenter ** 2 + ycenter ** 2 + zcenter ** 2)
        redshift_center = z_at_value(
            self.cosmo.comoving_distance,
            dc_center_i * u.Mpc,
            zmin=z.min() - 0.01,
            zmax=z.max() + 0.01,
        )
        return xcenter, ycenter, zcenter, dc_center_i, redshift_center

    def _halomass(self, radius, z_center):
        model = NFWProfile(self.cosmo, z_center, mdef=self.delta_c)
        hmass = model.halo_radius_to_halo_mass(radius)
        return hmass

    def _group_prop(self, id_groups, groups, xyz):
        id_groups = id_groups[id_groups > -1]
        xyzcenters = np.empty([len(id_groups), 3])
        dc_center = np.empty([len(id_groups)])
        hmass = np.empty([len(id_groups)])
        z_center = np.empty([len(id_groups)])
        for i in id_groups:
            mask = [groups == i]
            # halo radius
            radius = self._radius(self.ra[mask], self.dec[mask], self.z[mask])
            # halo center
            x, y, z, dc, z_cen = self._centers(xyz[mask], self.z[mask])
            xyzcenters[i, 0] = x
            xyzcenters[i, 1] = y
            xyzcenters[i, 2] = z
            dc_center[i] = dc
            z_center[i] = z_cen
            model = NFWProfile(self.cosmo, z_cen, mdef=self.delta_c)
            hmass[i] = model.halo_radius_to_halo_mass(radius)
        return xyzcenters, dc_center, z_center, radius, hmass

    def _bias(self, h0, mth, omega_m):
        pars = camb.CAMBparams()
        pars.set_cosmology(h0, ombh2=0.022, omch2=0.122)
        pars.set_dark_energy(w=-1.0)
        pars.InitPower.set_params(ns=0.965)
        pars.set_matter_power(redshifts=[0.0, 0.8], kmax=2.0)

        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=1, npoints=1000
        )
        bhm = bias.bias_at_M(mth, kh, pk, omega_m)
        return bhm

    def _dc_fog_corr(self, abs_mag, mag_threshold=-20.5, seedvalue=None):
        dcfogcorr = np.zeros(len(self.z))
        for i in Halo.labels_h_massive:
            sat_gal_mask = (GalInGroups.groups == i) * (
                abs_mag > mag_threshold
            )
            numgal = np.sum(sat_gal_mask)
            nfw = NFWProfile(self.cosmo, Halo.z_centers[i], mdef=self.delta_c)
            radial_positions_pos = nfw.mc_generate_nfw_radial_positions(
                num_pts=300000, halo_radius=Halo.radius[i], seed=seedvalue
            )
            radial_positions_neg = nfw.mc_generate_nfw_radial_positions(
                num_pts=300000, halo_radius=Halo.radius[i], seed=seedvalue
            )
            radial_positions_neg = -1 * radial_positions_neg
            radial_positions = np.r_[
                radial_positions_pos, radial_positions_neg
            ]
            dc = np.random.choice(radial_positions, size=numgal)
            dcfogcorr[sat_gal_mask] = Halo.dc_centers[i] + dc
        return dcfogcorr, Halo.dc_centers, Halo.radius, GalInGroups.groups

    def _z_fog_corr(self, dcfogcorr, abs_mag, mag_threshold=-20.5):
        zfogcorr = np.zeros(len(self.z))
        for i in Halo.labels_h_massive:
            sat_gal_mask = (GalInGroups.groups == i) * (
                abs_mag > mag_threshold
            )
            numgal = np.sum(sat_gal_mask)
            z_galaxies = np.zeros(numgal)
            dc_galaxies = dcfogcorr[sat_gal_mask]
            redshift = self.z[sat_gal_mask]
            for j in range(numgal):
                z_galaxies[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    dc_galaxies[j] * u.Mpc,
                    zmin=redshift.min() - 0.01,
                    zmax=redshift.max() + 0.01,
                )
            zfogcorr[sat_gal_mask] = z_galaxies
        return zfogcorr

    # ========================================================================
    # Public methods
    # ========================================================================

    def kaisercorr(self):
        """Corrects the Kaiser effect.

        Returns
        -------
        dckaisercorr : array_like
            Comoving distance to each galaxy after apply corrections for
            Kaiser effect. Array has the same lengh that the input
            array z.
        zkaisercorr : array_like
            Redshift of galaxies after apply corrections for Kaiser
            effect. Array has the same lengh that the input array z.

        """
        self.xyzcenters = self.xyzcenters[self.labelshmassive]
        inf = np.array(
            [
                self.xyzcenters[:, 0].min(),
                self.xyzcenters[:, 1].min(),
                self.xyzcenters[:, 2].min(),
            ]
        )
        sup = np.array(
            [
                self.xyzcenters[:, 0].max(),
                self.xyzcenters[:, 1].max(),
                self.xyzcenters[:, 2].max(),
            ]
        )
        rangeaxis = sup - inf
        maxaxis = np.argmax(rangeaxis)
        liminf = np.zeros((3))
        limsup = np.zeros((3))
        for i in range(3):
            if i == maxaxis:
                liminf[i] = inf[i] - 50
                limsup[i] = sup[i] + 50
            else:
                liminf[i] = (
                    inf[i] - (rangeaxis[maxaxis] + 100 - rangeaxis[i]) / 2
                )
                limsup[i] = (
                    sup[i] + (rangeaxis[maxaxis] + 100 - rangeaxis[i]) / 2
                )

        binesx = np.linspace(liminf[0], limsup[0], 1025)
        binesy = np.linspace(liminf[1], limsup[1], 1025)
        binesz = np.linspace(liminf[2], limsup[2], 1025)
        binnum = np.arange(0, 1024)
        xdist = pd.cut(self.xyzcenters[:, 0], bins=binesx, labels=binnum)
        ydist = pd.cut(self.xyzcenters[:, 1], bins=binesy, labels=binnum)
        zdist = pd.cut(self.xyzcenters[:, 2], bins=binesz, labels=binnum)
        self.valingrid = np.array(
            [
                np.array([xdist]),
                np.array([ydist]),
                np.array([zdist]),
            ]
        ).T

        bhm = self._bias(self.cosmo.H0, self.Mth, self.cosmo.Om0)

        x = np.arange(0, 1024)
        cube = np.array(np.meshgrid(x, x, x)).T.reshape(-1, 3)

        indexcube = np.zeros(1024 ** 3)
        for i in range(len(self.valingrid)):
            var = cube - self.valingrid[i]
            idcellsempty = np.where(
                (var[:, 0] == 0) & (var[:, 1] == 0) & (var[:, 2] == 0)
            )
            indexcube[idcellsempty] = self.mass[i]

        self.rho_h = np.sum(self.mass) / (1024 ** 3)
        self.delta = np.where(indexcube == 0, self.rho_h, indexcube)

        f = self.cosmo.Om0 ** 0.6 + 1 / 70 * self.cosmo.Ode0 * (
            1 + self.cosmo.Om0
        )

        v = np.fft.fft(self.cosmo.H0 * 1 * f * np.fft.fft(self.delta) / bhm)

        zkaisercorr = np.zeros((len(self.clustering.labels_)))

        for i in self.unique_elements:
            masc = [self.clustering.labels_ == i]
            zkaisercorr[masc] = (self.z[masc] - v[i] / const.c.value) / (
                1 + v[i] / const.c.value
            )

        dckaisercorr = self.cosmo.comoving_distance(zkaisercorr)

        return dckaisercorr, zkaisercorr

    #   reconstructed Kaiser space; based on correcting for FoG effect only
    def fogcorr(self, abs_mag, mag_threshold=-20.5, seedvalue=None):
        """Corrects the Finger of God effect only.

        Returns
        -------
        dcfogcorr : array_like
            Comoving distance to each galaxy after apply corrections for
            FoG effect. Array has the same lengh that the input
            array z.
        zfogcorr : array_like
            Redshift of galaxies after apply corrections for FoG
            effect. Array has the same lengh that the input array z.


        """
        dcfogcorr, dc_centers, radius, groups = self._dc_fog_corr(
            abs_mag, mag_threshold, seedvalue
        )
        zfogcorr = self._z_fog_corr(dcfogcorr, abs_mag, mag_threshold)
        dcfogcorr[dcfogcorr == 0] = self.cosmo.comoving_distance(
            self.z[dcfogcorr == 0]
        )
        zfogcorr[dcfogcorr == 0] = self.z[dcfogcorr == 0]
        return dcfogcorr, zfogcorr

    #    Re-real space reconstructed real space; based on correcting redshift
    #    space distortions
    def realspace(self):
        """Corrects Kaiser and FoG effect.

        Returns
        -------
        dc : array_like
            Comoving distance to each galaxy after apply corrections for
            Kaiser and FoG effects. Array has the same lengh that the input
            array z.
        zcorr : array_like
            Redshift of galaxies after apply corrections for Kaiser and FoG
            effects. Array has the same lengh that the input array z.

        """
        dcfogcorr, zfogcorr = self.fogcorr()
        dckaisercorr, zkaisercorr = self.kaisercorr()
        dc = dcfogcorr + dckaisercorr
        zcorr = np.zeros(len(self.z))
        for i in Halo.labels_h_massive:
            numgal = np.sum(GalInGroups.groups == i)
            z_galaxies = np.zeros(numgal)
            dc_galaxies = dc[GalInGroups.groups == i]
            redshift = self.z[GalInGroups.groups == i]
            for j in range(numgal):
                z_galaxies[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    dc_galaxies[j] * u.Mpc,
                    zmin=redshift.min() - 0.01,
                    zmax=redshift.max() + 0.01,
                )
            zcorr[GalInGroups.groups == i] = z_galaxies
        return dc, zcorr
