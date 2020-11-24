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
    """Store properties of dark matter halos."""

    xyzcenters = attr.ib()
    dc_centers = attr.ib()
    z_centers = attr.ib()
    radius = attr.ib()
    mass = attr.ib()
    labels_h_massive = attr.ib()


@attr.s(frozen=True)
class GalInGroup(object):
    """Store clustering results."""

    groups = attr.ib()
    id_groups = attr.ib()


# ============================================================================
# MAIN CLASS
# ============================================================================


@attr.s
class ReZSpace(object):
    """Real space reconstruction algorithm.

    This class have methods for corrects galaxy positions affected by Kaiser
    and Finger of God (FoG) effects.

    Parameters
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
    kaisercorr
        Corrects the Kaiser effect only.
    fogcorr
        Corrects the Finger of God effect only.
    realspace
        Corrects both effects (Kaiser and FoG).

    Notes:
    ------
    For the corrections is needed to determine the center, radius and mass of
    each dark matter halo. For this, we consider the geometric center, and the
    mass is calculated following a NFW profile (Navarro et al. 1997 [1]).
    The radius is estimated as in Merchán & Zandivarez (2005) [2].
    The threshold mass is used to determine which of the halos are massive.
    This is, massive halos are those whose mass are higher than the threshold
    mass.
    For the identification of halos we use the DBSCAN method of scikit-learn
    package, selecting the eps and min_samples parameters to obtanied the same
    galaxy groups of Zapata et al. (2009) [3] that have more of 150 members.

    References:
    -----------
    [1] Navarro J. F., Frenk C. S., White S. D., 1997, Apj, 490, 493
    [2] Merchán M., Zandivarez A., 2005, Apj, 630, 759
    [3] Zapata T., Pérez J., Padilla N., & Tissera P., 2009, MNRAS, 394, 2229


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

    def _dark_matter_halos(self):
        """Find properties of massive dark matter halos.

        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.
        Properties of halos: geometric center, comoving distance to center,
        redshift to center, radius of halo and mass of halo.

        """
        # cartesian coordinates for galaxies
        xyz = self._xyzcoordinates()
        # finding group of galaxies
        groups, id_groups = self._groups(xyz)
        # distance and redshifts to halo center
        # radius and mass of halo
        xyz_c, dc_c, z_c, rad, mass = self._group_prop(id_groups, groups, xyz)
        # selec massive halos
        labels_h_massive = np.where(mass > self.Mth)
        # store results of clustering
        galingroups = GalInGroup(groups, id_groups)
        # store properties of halos
        halos = Halo(xyz_c, dc_c, z_c, rad, mass, labels_h_massive)
        return halos, galingroups

    def _xyzcoordinates(self):
        """x, y, z cartesian coordinates to galaxies."""
        # comoving distance to galaxies
        dc = self.cosmo.comoving_distance(self.z)
        # set Ra and Dec in degrees. Comoving distance in Mpc
        c = SkyCoord(
            ra=np.array(self.ra) * u.degree,
            dec=np.array(self.dec) * u.degree,
            distance=np.array(dc) * u.mpc,
        )
        # creat an array with the results
        xyz = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
        return xyz

    def _groups(self, xyz):
        """Clustering of galaxies."""
        # set weights for clustering
        pesos = self.z * 100
        # clustering of galaxies
        clustering = DBSCAN(eps=1.2, min_samples=24)
        clustering.fit(xyz, sample_weight=pesos)
        # select only galaxies in groups
        unique_elements, counts_elements = np.unique(
            clustering.labels_, return_counts=True
        )
        unique_elements = unique_elements[unique_elements > -1]
        return clustering.labels_, unique_elements

    def _radius(self, ra, dec, z):
        """Radius of dark matter halos."""
        # number of galaxies
        galnum = len(ra)
        # comoving distance to galaxies
        dc = self.cosmo.comoving_distance(z)
        # prepare the coordinates for distance calculation
        c1 = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)
        c2 = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)
        # equation 6 of Merchán & Zandivarez (2005) [1]
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
        """Properties of centers of each halo."""
        # cartesian coordinates
        xcenter = np.mean(xyz[:, 0])
        ycenter = np.mean(xyz[:, 1])
        zcenter = np.mean(xyz[:, 2])
        # comoving distance to center
        dc_center_i = np.sqrt(xcenter ** 2 + ycenter ** 2 + zcenter ** 2)
        # redshift to center
        redshift_center = z_at_value(
            self.cosmo.comoving_distance,
            dc_center_i * u.Mpc,
            zmin=z.min() - 0.01,
            zmax=z.max() + 0.01,
        )
        return xcenter, ycenter, zcenter, dc_center_i, redshift_center

    def _halomass(self, radius, z_center):
        """Mass of halo."""
        # use a Navarro profile (Navarro et al. 1997) [1]
        model = NFWProfile(self.cosmo, z_center, mdef=self.delta_c)
        hmass = model.halo_radius_to_halo_mass(radius)
        return hmass

    def _group_prop(self, id_groups, groups, xyz):
        """Properties of halos."""
        # select only galaxies in groups
        id_groups = id_groups[id_groups > -1]
        # arrays to store return results
        xyzcenters = np.empty([len(id_groups), 3])
        dc_center = np.empty([len(id_groups)])
        hmass = np.empty([len(id_groups)])
        z_center = np.empty([len(id_groups)])
        radius = np.empty([len(id_groups)])
        # run for each group of galaxies
        for i in id_groups:
            mask = [groups == i]
            # halo radius
            radius[i] = self._radius(
                self.ra[mask], self.dec[mask], self.z[mask]
            )
            # halo center
            x, y, z, dc, z_cen = self._centers(xyz[mask], self.z[mask])
            xyzcenters[i, 0] = x
            xyzcenters[i, 1] = y
            xyzcenters[i, 2] = z
            dc_center[i] = dc
            z_center[i] = z_cen
            # halo mass
            model = NFWProfile(self.cosmo, z_cen, mdef=self.delta_c)
            hmass[i] = model.halo_radius_to_halo_mass(radius[i])
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
        """Corrected comoving distance."""
        # halo properties
        halos, galingroups = self._dark_matter_halos()
        # array to store return results
        dcfogcorr = np.zeros(len(self.z))
        # run for each massive halo
        for i in halos.labels_h_massive[0]:
            print(i)
            # select only galaxies with magnitudes over than -20.5
            sat_gal_mask = (galingroups.groups == i) * (
                abs_mag > mag_threshold
            )
            # number of galaxies for corrections
            numgal = np.sum(sat_gal_mask)
            # Monte Carlo simulation for distance
            nfw = NFWProfile(self.cosmo, halos.z_centers[i], mdef=self.delta_c)
            radial_positions_pos = nfw.mc_generate_nfw_radial_positions(
                num_pts=300000, halo_radius=halos.radius[i], seed=seedvalue
            )
            radial_positions_neg = nfw.mc_generate_nfw_radial_positions(
                num_pts=300000, halo_radius=halos.radius[i], seed=seedvalue
            )
            radial_positions_neg = -1 * radial_positions_neg
            radial_positions = np.r_[
                radial_positions_pos, radial_positions_neg
            ]
            # random choice of distance for each galaxy
            dc = np.random.choice(radial_positions, size=numgal)
            # combine Monte Carlo distance and distance to halo center
            dcfogcorr[sat_gal_mask] = halos.dc_centers[i] + dc
        return dcfogcorr, halos.dc_centers, halos.radius, galingroups.groups

    def _z_fog_corr(self, dcfogcorr, abs_mag, mag_threshold=-20.5):
        """Corrected redshift."""
        # halo properties
        halos, galingroups = self._dark_matter_halos()
        # array to store return results
        zfogcorr = np.zeros(len(self.z))
        # run for each massive halo
        for i in halos.labels_h_massive[0]:
            # select only galaxies with magnitudes over than -20.5
            sat_gal_mask = (galingroups.groups == i) * (
                abs_mag > mag_threshold
            )
            # number of galaxies for corrections
            numgal = np.sum(sat_gal_mask)
            z_galaxies = np.zeros(numgal)
            dc_galaxies = dcfogcorr[sat_gal_mask]
            redshift = self.z[sat_gal_mask]
            # corrected redshift of each galaxy
            # run for each galaxy
            for j in range(numgal):
                z_galaxies[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    dc_galaxies[j] * u.Mpc,
                    zmin=redshift.min() - 0.01,
                    zmax=redshift.max() + 0.01,
                )
            zfogcorr[sat_gal_mask] = z_galaxies
        return zfogcorr

    def _grid3d(self, centros, labels):
        centros = centros[labels]
        inf = np.array(
            [
                centros[:, 0].min(),
                centros[:, 1].min(),
                centros[:, 2].min(),
            ]
        )
        sup = np.array(
            [
                centros[:, 0].max(),
                centros[:, 1].max(),
                centros[:, 2].max(),
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
        xdist = pd.cut(centros[:, 0], bins=binesx, labels=binnum)
        ydist = pd.cut(centros[:, 1], bins=binesy, labels=binnum)
        zdist = pd.cut(centros[:, 2], bins=binesz, labels=binnum)
        valingrid = np.array(
            [
                np.array([xdist]),
                np.array([ydist]),
                np.array([zdist]),
            ]
        ).T
        return valingrid

    def _density(self, valingrid, mass, n):

        x = np.arange(0, n)
        cube = np.array(np.meshgrid(x, x, x)).T.reshape(-1, 3)
        indexcube = np.zeros(n ** 3)
        for i in range(len(valingrid)):
            var = cube - valingrid[i]
            idcellsempty = np.where(
                (var[:, 0] == 0) & (var[:, 1] == 0) & (var[:, 2] == 0)
            )
            indexcube[idcellsempty] = mass[i]

        rho_h = np.sum(mass) / (n ** 3)
        delta = np.where(indexcube == 0, rho_h, indexcube)
        return delta

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

        # Construct the 3d grid and return the cells in which the centers
        # of the halos are found
        valingrid = self._grid3d(Halo.xyzcenters, Halo.labels_h_massive)

        # Calculate bias
        bhm = self._bias(self.cosmo.H0, self.Mth, self.cosmo.Om0)

        # Calculate overdensity field
        delta = self._density(valingrid, Halo.mass, 1024)

        f = self.cosmo.Om0 ** 0.6 + 1 / 70 * self.cosmo.Ode0 * (
            1 + self.cosmo.Om0
        )

        ################################################################
        v = np.fft.fft(self.cosmo.H0 * 1 * f * np.fft.fft(delta) / bhm)

        zkaisercorr = np.zeros((len(self.clustering.labels_)))

        for i in self.unique_elements:
            masc = [self.clustering.labels_ == i]
            zkaisercorr[masc] = (self.z[masc] - v[i] / const.c.value) / (
                1 + v[i] / const.c.value
            )

        dckaisercorr = self.cosmo.comoving_distance(zkaisercorr)

        return dckaisercorr, zkaisercorr, v

    #   reconstructed Kaiser space; based on correcting for FoG effect only
    def fogcorr(self, abs_mag, mag_threshold=-20.5, seedvalue=None):
        """Corrects the Finger of God effect only.

        Parameters
        ----------
        abs_mag : array_like
            Absolute magnitudes of galaxies.
        mag_threshold : float, optional
            The threshold absolute magnitude that determines luminous center
            galaxies of each group. Default is -20.5.
        seedvalue : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        dcfogcorr : array_like
            Comoving distance to each galaxy after apply corrections for
            FoG effect. Array has the same lengh that the input
            array z.
        zfogcorr : array_like
            Redshift of galaxies after apply corrections for FoG
            effect. Array has the same lengh that the input array z.

        Example
        --------
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> dcfogcorr, zfogcorr = rzs.fogcorr(mags)

        Notes:
        ------
        This method use a Monte Carlo simulation to produce a NFW profile for
        the distances. The obtained distribution has 300000 values.
        Positions of luminous center galaxies are not corrected, only
        satellite galaxies. For this is use the magnitude threshold.


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
    def realspace(self, abs_mag, mag_threshold=-20.5, seedvalue=None):
        """Corrects Kaiser and FoG effect.

        Parameters
        ----------
        abs_mag : array_like
            Absolute magnitudes of galaxies.
        mag_threshold : float, optional
            The threshold absolute magnitude that determines luminous center
            galaxies of each group. Default is -20.5.
        seedvalue : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        dc : array_like
            Comoving distance to each galaxy after apply corrections for
            Kaiser and FoG effects. Array has the same lengh that the input
            array z.
        zcorr : array_like
            Redshift of galaxies after apply corrections for Kaiser and FoG
            effects. Array has the same lengh that the input array z.

        Notes:
        ------
        This method calls Kaisercorrr and Fogcorr methods, and combains
        their results.
        The method fogcorr use a Monte Carlo simulation to produce a NFW
        profile for the distances. The obtained distribution has 300000
        values.
        Positions of luminous center galaxies are not corrected for fogcorr,
        only satellite galaxies. For this is use the magnitude threshold.

        """
        halos, galingroups = self._dark_matter_halos()
        dcfogcorr, zfogcorr = self.fogcorr(abs_mag, mag_threshold, seedvalue)
        dckaisercorr, zkaisercorr = self.kaisercorr()
        dc = dcfogcorr + dckaisercorr
        zcorr = np.zeros(len(self.z))
        for i in Halo.labels_h_massive:
            numgal = np.sum(galingroups.groups == i)
            z_galaxies = np.zeros(numgal)
            dc_galaxies = dc[galingroups.groups == i]
            redshift = self.z[galingroups.groups == i]
            for j in range(numgal):
                z_galaxies[j] = z_at_value(
                    self.cosmo.comoving_distance,
                    dc_galaxies[j] * u.Mpc,
                    zmin=redshift.min() - 0.01,
                    zmax=redshift.max() + 0.01,
                )
            zcorr[galingroups.groups == i] = z_galaxies
        return dc, zcorr
