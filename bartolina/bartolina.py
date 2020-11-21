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
    cosmo : object
            Instance of an astropy cosmology. Default cosmology is
            LambdaCDM with H0=100, Om0=0.27, Ode0=0.73.
    Mth : float
          The threshold mass that determines massive halos in solar mass.
          Default is 10 ** 12.5.
    delta_c : string
              Overdensity constant. Default is "200m".

    Methods
    -------
    halos()
        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.
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
        self.clustering = DBSCAN(eps=1.2, min_samples=24)
        self.clustering.fit(xyz, sample_weight=pesos)
        unique_elements, counts_elements = np.unique(
            self.clustering.labels_, return_counts=True
        )
        self.unique_elements = unique_elements[unique_elements > -1]
        return self.clustering.labels_

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
            zmin=z.min(),
            zmax=z.max(),
        )
        return xcenter, ycenter, zcenter, dc_center_i, redshift_center

    def _halomass(self, radius, z_center):
        model = NFWProfile(self.cosmo, z_center, mdef=self.delta_c)
        hmass = model.halo_radius_to_halo_mass(radius)
        return hmass

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

    def halos(self):
        """Find massive dark matter halos.

        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.

        """
        # cartesian coordinates for galaxies
        xyz = self._xyzcoordinates()
        # finding group of galaxies
        self._groups(xyz)
        # mass and center, for each group
        self.xyzcenters = np.empty([len(self.unique_elements), 3])
        self.dc_center = np.empty([len(self.unique_elements)])
        self.hmass = np.empty([len(self.unique_elements)])
        for i in self.unique_elements:
            masksel = [self.clustering.labels_ == i]
            # halo radius
            radius = self._radius(
                self.ra[masksel], self.dec[masksel], self.z[masksel]
            )
            # halo center
            x, y, z, dc, z_cen = self._centers(xyz[masksel], self.z[masksel])
            self.xyzcenters[i, 0] = x
            self.xyzcenters[i, 0] = y
            self.xyzcenters[i, 0] = z
            self.dc_center[i] = dc
            # halo mass with NFW
            # NFW takes: halo radius, halo concentration parameter,
            # halo redshift, quantity consider, cosmology
            model = NFWProfile(self.cosmo, z_cen, mdef=dc)
            self.hmass[i] = model.halo_radius_to_halo_mass(radius)

        self.labelshmassive = np.where(self.mass > self.Mth)
        self.mass = self.mass[self.labelshmassive]

    def kaisercorr(self):
        """Corrects the Kaiser effect."""
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
    def fogcorr(self, seedvalue=None):
        """Corrects the Finger of God effect only."""
        dcfogcorr = np.zeros(len(self.clustering.labels_))
        zfogcorr = np.zeros(len(self.clustering.labels_))
        for i in self.labelshmassive[0]:

            numgal = np.sum(self.clustering.labels_ == i)
            c = 1 / 0.052
            r200 = 117
            rs = r200 / c
            n0 = (numgal / (4 * np.pi * rs ** 3)) * (
                1 / (np.log(1 + c) - c / (1 + c))
            )

            bins = np.linspace(0, 5000, 200)
            r = (bins[1:] - bins[:-1]) / 2 + bins[:-1]
            distr = n0 / ((r / rs) * (1 + r / rs) ** 2)

            distr = distr / np.sum(distr)
            ac = np.cumsum(distr)

            random = np.random.RandomState(seed=seedvalue)
            v3 = random(300000)
            v4 = np.zeros(300000)
            for j in range(len(ac) - 1):
                ind = np.where((v3 >= ac[j]) & (v3 < ac[j + 1]))
                v4[ind] = j + 1
            v5 = np.zeros(300000)
            for j in range(300000):
                v5[j] = (
                    bins[int(v4[j] + 1)] - bins[int(v4[j])]
                ) * np.random.random() + bins[int(v4[j])]

            self.dc = np.random.choice(v5, size=numgal)

            dcfogcorr[self.clustering.labels_ == i] = (
                self.dc_center[i] + self.dc
            )

            v6 = np.zeros(numgal)
            v7 = dcfogcorr[self.clustering.labels_ == i]
            redshift = self.z[self.clustering.labels_ == i]
            for j in range(numgal):
                v6[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    v7[j] * u.Mpc,
                    zmin=redshift.min() - 1,
                    zmax=redshift.max() + 1,
                )
            zfogcorr[self.clustering.labels_ == i] = v6

        dcfogcorr[dcfogcorr == 0] = self.cosmo.comoving_distance(
            self.z[dcfogcorr == 0]
        )
        zfogcorr[dcfogcorr == 0] = self.z[dcfogcorr == 0]

        return dcfogcorr, zfogcorr

    #    Re-real space reconstructed real space; based on correcting redshift
    #    space distortions
    def realspace(self):
        """Corrects Kaiser and FoG effect."""
        dcfogcorr, zfogcorr = self.fogcorr()
        dckaisercorr, zkaisercorr = self.kaisercorr()

        dc = dcfogcorr + dckaisercorr
        zcorr = np.zeros(len(self.clustering.labels_))
        for i in self.labelshmassive[0]:
            numgal = np.sum(self.clustering.labels_ == i)
            v6 = np.zeros(numgal)
            v7 = dc[self.clustering.labels_ == i]
            redshift = self.z[self.clustering.labels_ == i]
            for j in range(numgal):
                v6[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    v7[j] * u.Mpc,
                    zmin=redshift.min() - 1,
                    zmax=redshift.max() + 1,
                )
            zcorr[self.clustering.labels_ == i] = v6

        return dc, zcorr
