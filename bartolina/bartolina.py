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

import numpy as np

import pandas as pd

from sklearn.cluster import DBSCAN
from scipy import fftpack

import warnings
with warnings.catch_warnings(record=True) as w:
    from halotools.empirical_models import NFWProfile
    assert all(issubclass(wi.category, UserWarning) for wi in w)


# ============================================================================
# CONSTANTS
# ============================================================================


N_GRID_CELLS = 1024
N_MONTE_CARLO = 300000


# ============================================================================
# AUXILIARY CLASS
# ============================================================================


@attr.s(frozen=True)
class Halo(object):
    """Store properties of dark matter halos.

    Atributes
    ---------
    xyzcenters : ndarray
        Cartesian coordinates in Mpc to center of each halo.
    dc_centers : array_like
        Comoving distance to center of each halo.
    radius : array_like
        Radius of each halo in Mpc.
    mass : array_like
        Mass of each halo in solar mass.
    label_h_massive : array_like
          Label of halos with mass greater than the threshold mass.
              
    """

    xyzcenters = attr.ib()
    dc_centers = attr.ib()
    z_centers = attr.ib()
    radius = attr.ib()
    mass = attr.ib()
    labels_h_massive = attr.ib()


@attr.s(frozen=True)
class GalInGroup(object):
    """Store clustering results.

    Atributes
    ---------
    groups : array_like
        Cluster labels for each galaxy. Noisy samples are given the label -1.
        Is the same return than labels_ of sklearn.cluster.DBSCAN.
    id_group : array_like
        List of ids used in groups attribute.
    
    """

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
    dark_matter_halos
        Creates Halo and GalInGroup objects.
    xyzcoordinates
        Obtains cartesian coordinates of halos centers.    
    groups
        Finds groups of galaxies.
    radius
        Obtains radius of dark matter halos
    centers
        Finds halos centers.
    group_prop
        Obtaines properties of halos.
    bias
        DESCRIPCIÓN
    dc_fog_corr
        Corrects comoving distance only considering FoG effect.
    z_fog_corr
        Corrects redshift only considering FoG effect.
    grid3d
        DESCRIPCIÓN
    grid3d_axislim
        DESCRIPCIÓN    
    grid3d_gridlim
        DESCRIPCIÓN    
    grid3dcells
        DESCRIPCIÓN    
    density
        DESCRIPCIÓN    
    calcf
        DESCRIPCIÓN    
    zkaisercorr
        Corrects redshift only considering Kaiser effect.            
    kaisercorr
        Corrects the Kaiser effect only.
    fogcorr
        Corrects the Finger of God effect only.
    realspace
        Corrects both effects (Kaiser and FoG).

    Notes
    -----
    For the corrections is needed the center, radius and mass of
    each dark matter halo. For this, we consider the geometric center, and the
    mass is calculated following a NFW profile (Navarro et al. 1997 [1]).
    The radius is estimated as in Merchán & Zandivarez (2005) [2].
    The threshold mass is used to determine which of the halos are massive.
    This is, massive halos are those whose mass are higher than the threshold
    mass.
    For the identification of halos we use the DBSCAN method of scikit-learn
    package, selecting the eps and min_samples parameters to obtanied the same
    galaxy groups of Zapata et al. (2009) [3] that have more of 150 members.

    References
    ----------
    [1] Navarro J. F., Frenk C. S., White S. D., 1997, Apj, 490, 493
    [2] Merchán M., Zandivarez A., 2005, Apj, 630, 759
    [3] Zapata T., Pérez J., Padilla N., & Tissera P., 2009, MNRAS, 394, 2229


    """

    # User input params
    ra = attr.ib()
    dec = attr.ib()
    z = attr.ib()
    cosmo = attr.ib()
    @cosmo.default
    def _cosmo_default(self):
        return LambdaCDM(H0=100, Om0=0.27, Ode0=0.73)                    
    Mth = attr.ib(default=(10 ** 12.5))
    delta_c = attr.ib(default="200m")

    # ========================================================================
    # Internal methods
    # ========================================================================

    def dark_matter_halos(self):
        """Find properties of massive dark matter halos.

        Find massive dark matter halos and cartesian coordinates of his
        centers. Necesary for all the other methods.
        Properties of halos: geometric center, comoving distance to center,
        redshift to center, radius of halo and mass of halo.

        Returns
        -------
        halos : object
            This class store properties of dark matter halos i.e. mass, 
            radius, centers.
        galingroups :
            This class store clustering results.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> halos, galingroups = rzs.dark_matter_halos()

        Notes
        -----
        This method is separated into 3 small methods that perform each step
        separately (xyzcoordinates, groups and group_prop).

        """
        # cartesian coordinates for galaxies
        xyz = self.xyzcoordinates()
        # finding group of galaxies
        groups, id_groups = self.groups(xyz)
        # distance and redshifts to halo center
        # radius and mass of halo
        xyz_c, dc_c, z_c, rad, mass = self.group_prop(id_groups, groups, xyz)
        # selec massive halos
        labels_h_massive = np.where(mass > self.Mth)
        # store results of clustering
        galingroups = GalInGroup(groups, id_groups)
        # store properties of halos
        halos = Halo(xyz_c, dc_c, z_c, rad, mass, labels_h_massive)
        return halos, galingroups

    def xyzcoordinates(self):
        """Convert galaxies coordinates to Cartesian coordinates xyz.

        Returns
        -------
        xyz : ndarray
            Array containing Cartesian galaxies coordinates. Array has 3
            columns and the same length as the number of galaxies.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> xyz = rzs.xyzcoordinates()

        """

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

    def groups(self, xyz):
        """Galaxies clustering.

        Finds groups of galaxies.

        Returns
        -------
        clustering.labels_ : array_like
            Cluster labels for each galaxy. Noisy samples are given the label -1.
            Is the same return than labels_ of sklearn.cluster.DBSCAN.
        unique_elements : array_like
            List of ids used in clustering.labels_.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> xyz = rzs.xyzcoordinates()
        >>> groups, id_groups = rzs.groups(xyz)

        Notes
        -------
        To perform the calculation we have implemented sklearn.cluster.DBSCAN
        (API design for machine learning software: experiences from the
        scikit-learn project, Buitinck et al., 2013.)

        """

        # set weights for clustering
        weights = self.z * 100
        # clustering of galaxies
        clustering = DBSCAN(eps=1.2, min_samples=24)
        clustering.fit(xyz, sample_weight=weights)
        # select only galaxies in groups
        unique_elements, counts_elements = np.unique(
            clustering.labels_, return_counts=True
        )
        return clustering.labels_, unique_elements

    def radius(self, ra, dec, z):
        """Dark matter halos radius.

        Calculate the radius of the halos.

        Returns
        -------
        radius : array_like
            Radius of each halo in Mpc.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> xyz = rzs.xyzcoordinates()
        >>> groups, id_groups = rzs.groups(xyz)
        >>> mask = groups == 1
        >>> radius = rzs.radius(rzs.ra[mask], rzs.dec[mask], rzs.z[mask])

        """

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

    def centers(self, xyz, z):
        """Determine halos centers properties.

        Calculate Cartesian coordinates of halo centers, the halos comoving
        distances and the z-values of halo centers.

        Returns
        -------
        xcenter : array_like
            Cartesian coordinate in the x axis, in Mpc.
        ycenter : array_like
            Cartesian coordinate in the y axis, in Mpc.
        zcenter : array_like
            Cartesian coordinate in the z axis, in Mpc.
        dc_center_i : array_like
            Comoving distance to center of each halo.
        redshift_center : array_like
            Redshift to center of each halo.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> xyz = rzs.xyzcoordinates()
        >>> groups, id_groups = rzs.groups(xyz)
        >>> mask = groups == 1
        >>> x, y, z_value, dc, z_cen = rzs.centers(xyz[mask], rzs.z[mask])

        """

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

    #    def halomass(self, radius, z_center):
    #        """Mass of halo."""
    #        # use a Navarro profile (Navarro et al. 1997) [1]
    #        model = NFWProfile(self.cosmo, z_center, mdef=self.delta_c)
    #        hmass = model.halo_radius_to_halo_mass(radius)
    #        return hmass

    def group_prop(self, id_groups, groups, xyz):
        """Determine halos properties.

        Calculate Cartesian coordinates of halo centers, the halos comoving
        distances, the z-values of halo centers, halos radii and halos masses.

        Returns
        -------

        xyzcenters : ndarray
            Cartesian coordinates in Mpc to center of each halo.
        dc_center : array_like
            Comoving distance to center of each halo.
        z_center : array_like
            Cartesian coordinate in the z axis, in Mpc.
        radius : array_like
            Radius of each halo in Mpc.
        hmass : array_like
            Mass of each halo in solar mass.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> xyz = rzs.xyzcoordinates()
        >>> groups, id_groups = rzs.groups(xyz)
        >>> xyz_c, dc_c, z_c, rad, mass = rzs.group_prop(id_groups, groups,
                                                         xyz)

        Notes
        -----
        This method is separated into 2 small methods that perform each step
        separately (radius and centers).

        """

        # select only galaxies in groups
        galincluster = id_groups[id_groups > -1]
        # arrays to store results
        xyzcenters = np.empty([len(galincluster), 3])
        dc_center = np.empty([len(galincluster)])
        hmass = np.empty([len(galincluster)])
        z_center = np.empty([len(galincluster)])
        radius = np.empty([len(galincluster)])
        # run for each group of galaxies
        for i in galincluster:
            mask = groups == i
            # halo radius
            radius[i] = self.radius(
                self.ra[mask], self.dec[mask], self.z[mask]
            )
            # halo center
            x, y, z, dc, z_cen = self.centers(xyz[mask], self.z[mask])
            xyzcenters[i, 0] = x
            xyzcenters[i, 1] = y
            xyzcenters[i, 2] = z
            dc_center[i] = dc
            z_center[i] = z_cen
            # halo mass
            # use a Navarro profile (Navarro et al. 1997) [1]
            model = NFWProfile(self.cosmo, z_cen, mdef=self.delta_c)
            hmass[i] = model.halo_radius_to_halo_mass(radius[i])
        return xyzcenters, dc_center, z_center, radius, hmass

    def bias(self, h0, mth, omega_m):
        """Calculate halo bias function.

        Returns
        -------
        bhm : int
            Halo bias value from the entered cosmology and threshold mass.

        Example
        -------
        >>> import bartolina as bt
        >>> bhm = bt.bias(100, 10^12, 0.27)

        Notes
        -----
        To perform the calculation we have implemented cluster_toolkit package.

        """
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

    def dc_fog_corr(
        self,
        abs_mag,
        halos,
        galingroups,
        halo_centers,
        mag_threshold=-20.5,
        seedvalue=None,
    ):
        """Corrected comoving distance.

        Returns
        -------
        dcfogcorr : array_like
            Comoving distance only considering FoG effect.
        halo_centers : array_like
            Comoving distance to center of each halo.
        halos.radius : array_like
            Radius of each halo in Mpc.
        galingroups.groups : array_like
            Cluster labels for each galaxy. Noisy samples are given 
            the label -1. Is the same return than labels_ of 
            sklearn.cluster.DBSCAN.

        Example
        -------
        >>> dcfogcorr, dc_centers, radius, groups = self.dc_fog_corr(
            abs_mag,
            halos,
            galingroups,
            halos.dc_centers,
            mag_threshold,
            seedvalue)

        """
        # array to store return results
        dcfogcorr = np.zeros(len(self.z))
        # run for each massive halo
        for i in halos.labels_h_massive[0]:
            # select only galaxies with magnitudes over than -20.5
            sat_gal_mask = (galingroups.groups == i) * (
                abs_mag > mag_threshold
            )
            # number of galaxies for corrections
            numgal = np.sum(sat_gal_mask)
            # Monte Carlo simulation for distance
            nfw = NFWProfile(self.cosmo, halos.z_centers[i], mdef=self.delta_c)
            radial_positions_pos = nfw.mc_generate_nfw_radial_positions(
                num_pts=N_MONTE_CARLO, halo_radius=halos.radius[i], seed=seedvalue
            )
            radial_positions_neg = nfw.mc_generate_nfw_radial_positions(
                num_pts=N_MONTE_CARLO, halo_radius=halos.radius[i], seed=seedvalue
            )
            radial_positions_neg = -1 * radial_positions_neg
            radial_positions = np.r_[
                radial_positions_pos, radial_positions_neg
            ]
            # random choice of distance for each galaxy
            al = np.random.RandomState(seedvalue)
            dc = al.choice(radial_positions, size=numgal)
            # combine Monte Carlo distance and distance to halo center
            dcfogcorr[sat_gal_mask] = halo_centers[i] + dc
        return dcfogcorr, halo_centers, halos.radius, galingroups.groups

    def z_fog_corr(
        self, dcfogcorr, abs_mag, halos, galingroups, mag_threshold=-20.5
    ):
        """Corrected redshift.

        Returns
        -------
        zfogcorr: array_like
            DESCRIPCION

        Example
        -------


        Notes
        -----


        """
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

    def grid3d(self, centers, labels):
        """Create a cube box  whose linear size is chosen to be about 100 h^-1
        Mpc larger than then maximal scale of the survey volume among
        the three axes. Divide the box into 1024^3 grid cells and identify
        the cells in which the halos are found.

        Returns
        -------
        valingrid : array_like
            Numerical coordinates of the cells where the halos are found.
            Array has the same length that the input array centers.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> halos, galingroups = rzs.dark_matter_halos()
        >>> valingrid = rzs.grid3d(halos.xyzcenters, halos.labels_h_massive)

        Notes
        -----
        We have been based on the grids described in the works of Wang et al.
        2012 and Shi et al. 2016. This method is separated into 3 small
        methods that perform each step separately (grid3d_axislim,
                                                   grid3d_gridlim and
                                                   grid3dcells).

        """
        inf, sup = self.grid3d_axislim(centers, labels)

        liminf, limsup = self.grid3d_gridlim(inf, sup)

        valingrid = self.grid3dcells(liminf, limsup, centers, N_GRID_CELLS)

        return valingrid

    def grid3d_axislim(self, centers, labels):
        """Determines the minimum and maximum xyz coordinates in which the
        halos lie.

        Returns
        -------
        inf : array_like
            Minimum values of xyz. Array has 1 row and 3 columns.
        sup : array_like
            Maximum values of xyz. Array has 1 row and 3 columns.

        """
        centers = centers[labels]

        inf = np.array(
            [
                centers[:, 0].min(),
                centers[:, 1].min(),
                centers[:, 2].min(),
            ]
        )
        sup = np.array(
            [
                centers[:, 0].max(),
                centers[:, 1].max(),
                centers[:, 2].max(),
            ]
        )
        return inf, sup

    def grid3d_gridlim(self, inf, sup):
        """Determine the limits of the grid, which are chosen to be about
        100 h^-1 Mpc larger than then maximal scale of the survey volume among
        the three axes.

        Returns
        -------
        liminf : array_like
            Lower limits on xyz axes. Array has 1 row and 3 columns.
        limsup : array_like
            Upper limits on xyz axes. Array has 1 row and 3 columns.

        """
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
        return liminf, limsup

    def grid3dcells(self, liminf, limsup, centers, nbines):
        """Divide the box into 1024^3 grid cells and identify the cells in
        which the centers of halos are located.

        Returns
        -------
        valingrid : array_like
            Numerical coordinates of the cells where the halos are found.
            Array has the same length that the input array centers.

        """

        # Define cells
        binesx = np.linspace(liminf[0], limsup[0], nbines + 1)
        binesy = np.linspace(liminf[1], limsup[1], nbines + 1)
        binesz = np.linspace(liminf[2], limsup[2], nbines + 1)
        binnum = np.arange(0, nbines)
        # Define center's cells
        xdist = pd.cut(centers[:, 0], bins=binesx, labels=binnum)
        ydist = pd.cut(centers[:, 1], bins=binesy, labels=binnum)
        zdist = pd.cut(centers[:, 2], bins=binesz, labels=binnum)
        valingrid = np.array(
            [
                np.array([xdist]),
                np.array([ydist]),
                np.array([zdist]),
            ]
        ).T
        return valingrid

    def density(self, valingrid, mass, n):
        """Calculate the mass density in each cell.

        Returns
        -------
        delta : array_like
            Mass density in each cell. Array has the same length as the
            number of cells in the grid. In this case it is 1024 ^ 3.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> halos, galingroups = rzs.dark_matter_halos()
        >>> valingrid = rzs.grid3d(halos.xyzcenters, halos.labels_h_massive)
        >>> delta = rzs.density(valingrid, halos.mass, 1024)

        """
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

    def calcf(self, omegam, omegalambda):
        """Computes the approximation to the function f (\ omega).

        Returns
        -------
        f : int
            Value of f from the entered cosmology.

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> f = rzs.calcf(cosmo.Om0, cosmo.Ode0)

        or

        >>> import bartolina as bt
        >>> f = bt.calcf(0.27, 0.73)

        Notes
        -----
        The approach is based on Lahav et al. 1991.

        """
        f = omegam ** 0.6 + 1 / 70 * omegalambda * (1 + omegam)
        return f

    def zkaisercorr(self, zcenters, velocity):
        zkaisercorr = (zcenters - velocity / const.c.value) / (
            1 + velocity / const.c.value
        )
        return zkaisercorr

    # ========================================================================
    # Public methods
    # ========================================================================

    # Reconstructed FoG space; based on correcting for Kaiser effect only
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

        Example
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> dckaisercorr, zkaisercorr = rzs.kaisercorr()

        Notes
        -----
        We embed the survey volume in a cube box whose linear size is chosen
        to be about 100 h^-1 Mpc larger than then maximal scale of the survey
        volume among the three axes. We compute the overdensity field of
        groups and based on Colombi, Chodorowski & Teyssier, 2007 we compute
        the peculiar velocity corrected. Finally we found the cosmological
        redshift of each group.

        """
        halos, galingroups = self.dark_matter_halos()

        # Construct the 3d grid and return the cells in which the centers
        # of the halos are found
        valingrid = self.grid3d(halos.xyzcenters, halos.labels_h_massive)

        # Calculate bias
        bhm = self.bias(self.cosmo.H0.value, self.Mth, self.cosmo.Om0)

        # Calculate overdensity field
        delta = self.density(valingrid, halos.mass, N_GRID_CELLS)

        f = self.calcf(self.cosmo.Om0, self.cosmo.Ode0)

        inf, sup = self._grid3d_axislim(halos.xyzcenters, halos.labels_h_massive)
        liminf, limsup = self._grid3d_gridlim(inf, sup)
        kx0 = fftpack.fftfreq(N_GRID_CELLS, d=(limsup[0]-liminf[0])/N_GRID_CELLS)
        ky0 = fftpack.fftfreq(N_GRID_CELLS, d=(limsup[1]-liminf[1])/N_GRID_CELLS)
        kz0 = fftpack.fftfreq(N_GRID_CELLS, d=(limsup[2]-liminf[2])/N_GRID_CELLS)
        kx, ky, kz = np.meshgrid(kx0, ky0, kz0)
        k2 = kx**2 + ky**2 + kz**2
        vx = fftpack.ifft(self.cosmo.H0 * 1 * f * (1j*kx/k2) * fftpack.fft(delta) / bhm)
        vy = fftpack.ifft(self.cosmo.H0 * 1 * f * (1j*ky/k2) * fftpack.fft(delta) / bhm)
        vz = fftpack.ifft(self.cosmo.H0 * 1 * f * (1j*kz/k2) * fftpack.fft(delta) / bhm)        

        v= np.sqrt(vx**2 + vy**2 + vz**2)
        
        zkaisercorr = self.zkaisercorr(halos.z_centers, v)

        dckaisercorr = self.cosmo.comoving_distance(zkaisercorr)

        return dckaisercorr, zkaisercorr

    # Reconstructed Kaiser space; based on correcting for FoG effect only
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
            Random number seed used in the Monte Carlo realization.ismo error.
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
        -------
        >>> import bartolina as bt
        >>> rzs = bt.ReZSpace(ra, dec, z)
        >>> dcfogcorr, zfogcorr = rzs.fogcorr(mags)

        Notes
        -----
        This method use a Monte Carlo simulation to produce a NFW profile for
        the distances. The obtained distribution has 300000 values.
        Positions of luminous center galaxies are not corrected, only
        satellite galaxies. For this is use the magnitude threshold.


        """
        # halo properties
        halos, galingroups = self.dark_matter_halos()
        dcfogcorr, dc_centers, radius, groups = self.dc_fog_corr(
            abs_mag,
            halos,
            galingroups,
            halos.dc_centers,
            mag_threshold,
            seedvalue,
        )
        zfogcorr = self.z_fog_corr(
            dcfogcorr, abs_mag, halos, galingroups, mag_threshold
        )
        dcfogcorr[dcfogcorr == 0] = self.cosmo.comoving_distance(
            self.z[dcfogcorr == 0]
        )
        zfogcorr[zfogcorr == 0] = self.z[zfogcorr == 0]
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

        Notes
        -----
        This method calls Kaisercorrr and Fogcorr methods, and combains
        their results.
        The method fogcorr use a Monte Carlo simulation to produce a NFW
        profile for the distances. The obtained distribution has 300000
        values.
        Positions of luminous center galaxies are not corrected for fogcorr,
        only satellite galaxies. For this is use the magnitude threshold.

        """
        # array to store return results
        dc = np.zeros(len(self.z))
        # properties of halos
        halos, galingroups = self.dark_matter_halos()
        # Kaiser correction with kaisercorr method
        dckaisercorr, zkaisercorr = self.kaisercorr()
        # FoG correction with fogcorr method
        dccorr, dc_centers, radius, groups = self.dc_fog_corr(
            abs_mag, halos, galingroups, dckaisercorr, mag_threshold, seedvalue
        )
        zcorr = self.z_fog_corr(
            dccorr, abs_mag, halos, galingroups, mag_threshold
        )
        dccorr[dccorr == 0] = self.cosmo.comoving_distance(self.z[dccorr == 0])
        zcorr[zcorr == 0] = self.z[zcorr == 0]
        # corrected redshift of each galaxy
        # run for each massive halo
        return dc, zcorr
