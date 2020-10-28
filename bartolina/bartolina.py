# This file is part of the
#   Bartolina Project (https://github.com/exiliadadelsur/Bartolina).
# Copyright (c) 2020 Noelia Perez
# License: MIT
#   Full Text: https://github.com/exiliadadelsur/Bartolina/blob/master/LICENSE

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

    ra = attr.ib()
    dec = attr.ib()
    z = attr.ib()
    cosmo = attr.ib(default=LambdaCDM(H0=100, Om0=0.27, Ode0=0.73))
    Mth = attr.ib(default=(10 ** 12.5))

    def Halos(self):

        dc = self.cosmo.comoving_distance(self.z)
        c = SkyCoord(
            ra=np.array(self.ra) * u.degree,
            dec=np.array(self.dec) * u.degree,
            distance=np.array(dc) * u.mpc,
        )
        # coordinates transform
        xyz = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
        pesos = 1 + np.arctan(self.z / 0.050)
        self.clustering = DBSCAN(eps=3, min_samples=130)
        self.clustering.fit(xyz, sample_weight=pesos)

        unique_elements, counts_elements = np.unique(
            self.clustering.labels_, return_counts=True
        )

        self.unique_elements = unique_elements[unique_elements > -1]
        self.xyzcentros = np.empty([len(unique_elements), 3])
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
            dc_centro = np.sqrt(
                self.xyzcentros[i, 0] ** 2
                + self.xyzcentros[i, 1] ** 2
                + self.xyzcentros[i, 2] ** 2
            )
            redshift = self.z[self.clustering.labels_ == i]
            z_centro = z_at_value(
                self.cosmo.comoving_distance,
                dc_centro * u.Mpc,
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

    # reconstructed FoG space; based on correcting for Kaiser effect only

    def Kaisercorr(self):

        self.xyzcentros = self.xyzcentros[self.labelshmassive]
        limxsup = self.xyzcentros[:, 0].max()
        limysup = self.xyzcentros[:, 1].max()
        limzsup = self.xyzcentros[:, 2].max()
        limsup = np.max(np.array([limxsup, limysup, limzsup])) + 0.001
        limxmin = self.xyzcentros[:, 0].min()
        limymin = self.xyzcentros[:, 1].min()
        limzmin = self.xyzcentros[:, 2].min()
        liminf = np.min(np.array([limxmin, limymin, limzmin])) - 0.001
        bines = np.linspace(liminf, limsup, 1024)
        binnum = np.arange(0, 1023)
        xdist = pd.cut(self.xyzcentros[:, 0], bins=bines, labels=binnum)
        ydist = pd.cut(self.xyzcentros[:, 1], bins=bines, labels=binnum)
        zdist = pd.cut(self.xyzcentros[:, 2], bins=bines, labels=binnum)
        self.valingrid = np.array(
            [
                np.array([xdist]),
                np.array([ydist]),
                np.array([zdist]),
            ]
        ).T

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
    def FoGcorr(self):

        for i in self.labelshmassive[0]:
            numgal = len(self.clustering.labels_ == (i + 1))
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

            v3 = np.random.random(300000)
            v4 = np.zeros(300000)
            for i in range(len(ac) - 1):
                ind = np.where((v3 >= ac[i]) & (v3 < ac[i + 1]))
                v4[ind] = i + 1
            v5 = np.zeros(300000)
            for i in range(300000):
                v5[i] = (
                    bins[int(v4[i] + 1)] - bins[int(v4[i])]
                ) * np.random.random() + bins[int(v4[i])]

    # self.dc=np.random.choice(Sim['DistrVal'],size=numgal)

    #       dccorr = dc_centro + distr

    #    Re-real space reconstructed real space; based on correcting redshift
    #    space distortions
    def RealSpace(self):
        pass
        # llama a ReKaiserSpace + ReFoGSpace


#       rcomovingk = ReKaiserSpace()
#       rcomovingf = ReFoGSpace()
#       return rcomovingk + rcomovingf
