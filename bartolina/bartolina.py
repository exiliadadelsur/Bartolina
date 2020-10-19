import numpy as np
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

    def mclustering(self):
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

        unique_elements = unique_elements[unique_elements > -1]
        xyzcentros = np.empty([len(unique_elements), 3])
        hmass = np.empty([len(unique_elements)])
        for i in unique_elements:
            v1 = xyz[self.clustering.labels_ == i]
            xyzcentros[i, 0] = np.mean(v1[:, 0])
            xyzcentros[i, 1] = np.mean(v1[:, 1])
            xyzcentros[i, 2] = np.mean(v1[:, 2])
            xradio = np.std(xyz[:, 0])
            yradio = np.std(xyz[:, 1])
            zradio = np.std(xyz[:, 2])
            radio = np.sqrt(xradio ** 2 + yradio ** 2 + zradio ** 2)
            dc_centro = np.sqrt(
                xyzcentros[i, 0] ** 2
                + xyzcentros[i, 1] ** 2
                + xyzcentros[i, 2] ** 2
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

        self.hmass = hmass[hmass > self.Mth]

    # reconstructed Kaiser space; based on correcting for FOG effect only


#    def ReKaiserSpace(self):

#       Grillado 3D

#       Halo bias
#       calcular k y P_linear con CAMB
#       cosmología
#       pars = camb.CAMBparams()
#       pars.set_cosmology(self.H0, ombh2=0.022, omch2=0.122)
#       pars.set_dark_energy(w=-1.0)
#       pars.InitPower.set_params(ns=0.965)
#       pars.set_matter_power(redshifts=[0.0, 0.8], kmax=2.0)
#       calcula plineal
#       pars.NonLinear = model.NonLinear_none
#       results = camb.get_results(pars)
#       kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
#                                                     npoints=1000)
#       bhm = bias.bias_at_M(self.Mth, kh, pk, self.omega_m)
#       Mass density contrast
#       deltah =
#       Calculo de f
#       f = self.omega_m ** 0.6 + 1 / 70 * self.omega_lambda *
#       (1 + self.omega_m)
#       ReKaiserZ
#       zk = (self.z - v/const.c) / 1 + v/const.c
#       Comoving distance
#       rcomovingk = calculo de distancia comoving a partir de zk
#       return #rcomovingk

#   reconstructed FOG space; based on correcting for Kaiser effect only
#   def ReFoGSpace(self):
#       return rcomovingf

#    Re-real space reconstructed real space; based on correcting redshift
#    space distortions
#    def ReRealSpace(self):
#       llama a ReKaiserSpace + ReFoGSpace
#       rcomovingk = ReKaiserSpace()
#       rcomovingf = ReFoGSpace()
#       return rcomovingk + rcomovingf
