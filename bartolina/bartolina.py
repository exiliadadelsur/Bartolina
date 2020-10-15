import numpy as np
from astropy.cosmology import LambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord

from sklearn.cluster import DBSCAN

# from cluster_toolkit import bias
# import camb
# from camb import model

import attr


@attr.s
class ReZSpace(object):

    ra = attr.ib()
    dec = attr.ib()
    z = attr.ib()
    H0 = attr.ib()
    omega_m = attr.ib()
    omega_lambda = attr.ib()
    Mth = attr.ib(default=(10 ** 12.5))

    def mclustering(self):
        cosmo = LambdaCDM(H0=self.H0, Om0=self.omega_m, Ode0=self.omega_lambda)
        dc = cosmo.comoving_distance(self.z)
        c = SkyCoord(
            ra=np.array(self.ra) * u.degree,
            dec=np.array(self.dec) * u.degree,
            distance=np.array(dc) * u.mpc,
        )
        xyz = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
        pesos = 1 + np.arctan(self.z / 0.050)
        self.clustering = DBSCAN(eps=3, min_samples=130)
        self.clustering.fit(xyz, sample_weight=pesos)

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
