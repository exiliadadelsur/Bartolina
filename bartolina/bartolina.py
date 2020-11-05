# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Perez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

"""Bartolina : real space reconstruction algorithm for redshift."""

import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM, z_at_value
from astropy import units as u
from astropy.coordinates import SkyCoord

from sklearn.cluster import DBSCAN
from NFW.nfw import NFW


# from cluster_toolkit import bias
# import camb
# from camb import model

import attr


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
    cosmo : astropy class
            Astropy class that describe the cosmology to use.
    Mth : float
          The threshold mass that determines massive halos.

    Methods
    -------
    Halos()
        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.
    Kaisercorr()
        Corrects the Kaiser effect only.
    FoGcorr()
        Corrects the Finger of God effect only.
    RealSpace()
        Corrects both effects (Kaiser and FoG).

    """

    ra = attr.ib()
    dec = attr.ib()
    z = attr.ib()
    cosmo = attr.ib(default=LambdaCDM(H0=100, Om0=0.27, Ode0=0.73))
    Mth = attr.ib(default=(10 ** 12.5))

    def Halos(self):
        """Find massive dark matter halos.

        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.

        """
        dc = self.cosmo.comoving_distance(self.z)
        c = SkyCoord(
            ra=np.array(self.ra) * u.degree,
            dec=np.array(self.dec) * u.degree,
            distance=np.array(dc) * u.mpc,
        )
        xyz = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
        pesos = 1 + np.arctan(self.z / 0.050)
        self.clustering = DBSCAN(eps=3, min_samples=130)
        self.clustering.fit(xyz, sample_weight=pesos)

        unique_elements, counts_elements = np.unique(
            self.clustering.labels_, return_counts=True
        )

        self.unique_elements = unique_elements[unique_elements > -1]
        self.xyzcentros = np.empty([len(unique_elements), 3])
        self.dc_centro = np.empty([len(unique_elements)])
        hmass = np.empty([len(unique_elements)])
        for i in unique_elements:
            v1 = xyz[self.clustering.labels_ == i]
            self.xyzcentros[i, 0] = np.mean(v1[:, 0])
            self.xyzcentros[i, 1] = np.mean(v1[:, 1])
            self.xyzcentros[i, 2] = np.mean(v1[:, 2])
            xradio = np.std(xyz[:, 0])
            yradio = np.std(xyz[:, 1])
            zradio = np.std(xyz[:, 2])
            radio = np.sqrt(xradio ** 2 + yradio ** 2 + zradio ** 2)
            self.dc_centro[i] = np.sqrt(
                self.xyzcentros[i, 0] ** 2
                + self.xyzcentros[i, 1] ** 2
                + self.xyzcentros[i, 2] ** 2
            )
            redshift = self.z[self.clustering.labels_ == i]
            z_centro = z_at_value(
                self.cosmo.comoving_distance,
                self.dc_centro[i] * u.Mpc,
                zmin=redshift.min(),
                zmax=redshift.max(),
            )
            cparam = 4
            nfw = NFW(
                radio, 4, z_centro, size_type="radius", cosmology=self.cosmo
            )
            r200 = cparam * radio
            hmass[i] = nfw.mass(r200).value

        self.labelshmassive = np.where(hmass > self.Mth)

    def Kaisercorr(self):
        """Corrects the Kaiser effect."""
        self.xyzcentros = self.xyzcentros[self.labelshmassive]
        inf = np.array(
            [
                self.xyzcentros[:, 0].min(),
                self.xyzcentros[:, 1].min(),
                self.xyzcentros[:, 2].min(),
            ]
        )
        sup = np.array(
            [
                self.xyzcentros[:, 0].max(),
                self.xyzcentros[:, 1].max(),
                self.xyzcentros[:, 2].max(),
            ]
        )
        rangeaxis = sup - inf
        maxaxis = np.argmax(rangeaxis)
        liminf = np.empty((3))
        limsup = np.empty((3))
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

        binesx = np.linspace(liminf[0], limsup[0], 1024)
        binesy = np.linspace(liminf[1], limsup[1], 1024)
        binesz = np.linspace(liminf[2], limsup[2], 1024)
        binnum = np.arange(0, 1023)
        xdist = pd.cut(self.xyzcentros[:, 0], bins=binesx, labels=binnum)
        ydist = pd.cut(self.xyzcentros[:, 1], bins=binesy, labels=binnum)
        zdist = pd.cut(self.xyzcentros[:, 2], bins=binesz, labels=binnum)
        self.valingrid = np.array(
            [
                np.array([xdist]),
                np.array([ydist]),
                np.array([zdist]),
            ]
        ).T

        self.rho_h = len(self.xyzcentros) / 1024 ** 3

        # Halo bias
        # calcular k y P_linear con CAMB
        # cosmología
        # pars = camb.CAMBparams()
        # pars.set_cosmology(self.H0, ombh2=0.022, omch2=0.122)
        # pars.set_dark_energy(w=-1.0)
        # pars.InitPower.set_params(ns=0.965)
        # pars.set_matter_power(redshifts=[0.0, 0.8], kmax=2.0)
        # calcula plineal
        # pars.NonLinear = model.NonLinear_none
        # results = camb.get_results(pars)
        # kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
        #                                                 npoints=1000)
        # bhm = bias.bias_at_M(self.Mth, kh, pk, self.omega_m)
        # Mass density contrast
        # deltah =
        # Calculo de f
        # f = self.omega_m ** 0.6 + 1 / 70 * self.omega_lambda *
        # (1 + self.omega_m)
        # ReKaiserZ
        # zk = (self.z - v/const.c) / 1 + v/const.c
        # Comoving distance
        # rcomovingk = calculo de distancia comoving a partir de zk
        # return rcomovingk

    #   reconstructed Kaiser space; based on correcting for FoG effect only
    def FoGcorr(self, seedvalue=None):
        """Corrects the Finger of God effect only."""
        dcFoGcorr = np.zeros(len(self.clustering.labels_))
        zFoGcorr = np.zeros(len(self.clustering.labels_))
        for i in self.labelshmassive[0]:

            numgal = np.sum(self.clustering.labels_ == i)
            c = 1 / 0.052
            r200 = 117
            rs = r200 / c
            n0 = (numgal / (4 * np.pi * rs ** 3)) * (
                1 / (np.log(1 + c) - c / (1 + c))
            )

            bins = np.linspace(0, 5000, 200)
            r = (
                (
                    bins[
                        1:,
                    ]
                    - bins[
                        :-1,
                    ]
                )
                / 2
            ) + bins[:-1]
            distr = n0 / ((r / rs) * (1 + r / rs) ** 2)

            distr = distr / np.sum(distr)
            ac = np.cumsum(distr)
            if seedvalue is not None:
                np.random.seed(seedvalue)
            v3 = np.random.random(300000)
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
            dcFoGcorr[self.clustering.labels_ == i] = (
                self.dc_centro[i] + self.dc
            )

            v6 = np.zeros(numgal)
            v7 = dcFoGcorr[self.clustering.labels_ == i]
            redshift = self.z[self.clustering.labels_ == i]
            for j in range(numgal):
                v6[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    v7[j] * u.Mpc,
                    zmin=redshift.min() - 1,
                    zmax=redshift.max() + 1,
                )
            zFoGcorr[self.clustering.labels_ == i] = v6
        dcFoGcorr[dcFoGcorr == 0] = self.cosmo.comoving_distance(
            self.z[dcFoGcorr == 0]
        )
        zFoGcorr[dcFoGcorr == 0] = self.z[dcFoGcorr == 0]

        return dcFoGcorr, zFoGcorr

    #    Re-real space reconstructed real space; based on correcting redshift
    #    space distortions
    def RealSpace(self):
        """Corrects Kaiser and FoG effect."""
        dcFoGcorr, zFoGcorr = self.FoGcorr()
        dcKaisercorr, zKaisercorr = self.Kaisercorr()

        dc = dcFoGcorr + dcKaisercorr
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
