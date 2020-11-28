"""This module defines analysis routines and associated machinery

Notes
-----
Without laminate details like layup and ply properties, this module is limited to max strain analysis. However, with
a third party CLPT package , the parent class `Analysis` can be used to extend this module to check max stress, Tsai-Wu,
Tsai-Hill, etc.

"""
import numpy as np
import bjsfm.lekhnitskii as lk
from bjsfm.plotting import plot_stress


class Analysis:
    """Parent class for analysis sub-classes

    Parameters
    ----------
    thickness : float
        plate thickness
    diameter : float
        hole diameter
    a_matrix : array_like
        2D 3x3 A-matrix from CLPT

    Attributes
    ----------
    t : float
        plate thickness
    r : float
        hole radius
    a : ndarray
        2D 3x3 A-matrix from CLPT
    a_inv : ndarray
        2D 3x3 Inverse A-matrix from CLPT

    """

    def __init__(self, a_matrix, thickness, diameter):
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix, dtype=float)
        self.a_inv = np.linalg.inv(self.a)

    def _loaded(self, bearing):
        """Lekhnitskii's loaded hole solution

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array [Px, Py]

        Returns
        -------
        bjsfm.lekhnitskii.LoadedHole

        """
        d = self.r*2
        t = self.t
        a_inv = self.a_inv
        bearing = np.array(bearing, dtype=float)
        p, theta = self.bearing_angle(bearing)
        return lk.LoadedHole(p, d, t, a_inv, theta=theta)

    def _unloaded(self, bearing, bypass, w=0.):
        """Lekhnitskii's unloaded hole solution

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array [Px, Py]
        bypass : array_like
            1D 1x3 array [Nx, Ny, Nxy]
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)

        Returns
        -------
        bjsfm.lekhnitskii.UnloadedHole

        """
        d = self.r*2
        t = self.t
        a_inv = self.a_inv
        bearing = np.array(bearing, dtype=float)
        bypass = np.array(bypass, dtype=float)
        p, theta = self.bearing_angle(bearing)
        if w:  # DeJong correction for finite width
            brg_dir_bypass = lk.rotate_plane_stress(bypass, angle=theta)
            sign = np.sign(brg_dir_bypass[0]) if abs(brg_dir_bypass[0]) > 0 else 1.
            bypass += lk.rotate_plane_stress(np.array([p/(2*w)*sign, 0., 0.]), angle=-theta)
        return lk.UnloadedHole(bypass, d, t, a_inv)

    def polar_points(self, rc=0., num=100):
        """Calculates r, theta points

        Parameters
        ----------
        rc : float, default 0.
            distance away from hole
        num: int, default 100
            number of points

        Returns
        -------
        r, theta : ndarray
            1D arrays, polar r, theta locations

        """
        r = np.array([self.r + rc] * num)
        theta = np.linspace(0, 2*np.pi, num=num, endpoint=False)
        return r, theta

    def xy_points(self, rc=0., num=100):
        """Calculates x, y points

        Parameters
        ----------
        rc : float, default 0.
            distance away from hole
        num: int, default 100
            number of points

        Returns
        -------
        x, y : ndarray
            1D arrays, cartesian x, y locations

        """
        r, theta = self.polar_points(rc=rc, num=num)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    @staticmethod
    def bearing_angle(bearing):
        """Calculates bearing load and angle

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array [Px, Py]

        Returns
        -------
        p, theta : float
            bearing load (p) and angle (theta)

        """
        p = np.sqrt(np.sum(np.square(bearing)))
        if p == 0:
            return 0., 0.
        theta = np.arccos(np.array([1, 0]).dot(bearing) / p)
        theta = theta*np.sign(bearing[1]) if bearing[1] else theta
        return p, theta

    def stresses(self, bearing, bypass, rc=0., num=100, w=0.):
        """ Calculate stresses

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)

        Returns
        -------
        ndarray
            2D numx3 array of plate stresses

        """
        x, y = self.xy_points(rc=rc, num=num)
        byp = self._unloaded(bearing, bypass, w=w)
        brg = self._loaded(bearing)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        return byp_stress + brg_stress

    def strains(self, bearing, bypass, rc=0., num=100, w=0.):
        """ Calculate strains

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)

        Returns
        -------
        ndarray
            2D numx3 array of plate strains

        """
        stresses = self.stresses(bearing, bypass, rc=rc, num=num, w=w)
        strains = self.a_inv @ stresses.T*self.t
        return strains.T

    def plot_stress(self, bearing, bypass, w=0., comp=0, rnum=100, tnum=100, axes=None,
                xbounds=None, ybounds=None, cmap='jet', cmin=None, cmax=None):
        """ Plots stresses

        Notes
        -----
        colormap options can be found at
            https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)
        comp : {0, 1, 2}, default 0
            stress component
        rnum : int, default 100
            number of points to plot along radius
        tnum : int, default 100
            number of points to plot along circumference
        axes : matplotlib.axes, optional
            a custom axes to plot on
        xbounds : tuple of int, optional
            (x0, x1) x-axis bounds, default 6*radius
        ybounds : tuple of int, optional
            (y0, y1) y-axis bounds, default 6*radius
        cmap : str, optional
            name of any colormap name from matplotlib.pyplot
        cmin : float, optional
            minimum value for colormap
        cmax : float, optional
            maximum value for colormap

        """
        plot_stress(lk_1=self._unloaded(bearing, bypass, w=w), lk_2=self._loaded(bearing), comp=comp, rnum=rnum,
                    tnum=tnum, axes=axes, xbounds=xbounds, ybounds=ybounds, cmap=cmap, cmin=cmin, cmax=cmax)


class MaxStrain(Analysis):
    """A class for analyzing joint failure using max strain failure theory

    Notes
    -----
    This class only supports four angle laminates, and is setup for laminates with 0, 45, -45 and 90 degree plies.

    Parameters
    ----------
    thickness : float
        plate thickness
    diameter : float
        hole diameter
    a_matrix : array_like
        2D 3x3 inverse A-matrix from CLPT
    et0 : float
        tension strain allowable in 0 deg direction
    et90 : float
        tension strain allowable in 90 deg direction
    et45 : float
        tension strain allowable in 45 deg direction
    etn45 : float
        tension strain allowable in -45 deg direction
    ec0 : float
        compression strain allowable in 0 deg direction
    ec90 : float
        compression strain allowable in 90 deg direction
    ec45 : float
        compression strain allowable in 45 deg direction
    ecn45 : float
        compression strain allowable in -45 deg direction
    es0 : float
        shear strain allowable in 0 deg direction
    es90 : float
        shear strain allowable in 90 deg direction
    es45 : float
        shear strain allowable in 45 deg direction
    esn45 : float
        shear strain allowable in -45 deg direction

    Attributes
    ----------
    t : float
        plate thickness
    r : float
        hole radius
    a : ndarray
        2D 3x3 A-matrix from CLPT
    a_inv : ndarray
        2D 3x3 Inverse A-matrix from CLPT
    et0 : float
        plate tension strain allowable in 0 deg direction
    et90 : float
        plate tension strain allowable in 90 deg direction
    et45 : float
        plate tension strain allowable in 45 deg direction
    etn45 : float
        plate tension strain allowable in -45 deg direction
    ec0 : float
        plate compression strain allowable in 0 deg direction
    ec90 : float
        plate compression strain allowable in 90 deg direction
    ec45 : float
        plate compression strain allowable in 45 deg direction
    ecn45 : float
        plate compression strain allowable in -45 deg direction
    es0 : float
        plate shear strain allowable in 0 deg direction
    es90 : float
        plate shear strain allowable in 90 deg direction
    es45 : float
        plate shear strain allowable in 45 deg direction
    esn45 : float
        plate shear strain allowable in -45 deg direction

    """

    def __init__(self, a_matrix, thickness, diameter, et0=None, et90=None, et45=None, etn45=None, ec0=None, ec90=None,
                 ec45=None, ecn45=None, es0=None, es90=None, es45=None, esn45=None):
        super(MaxStrain, self).__init__(a_matrix, thickness, diameter)
        # TODO: convert to dictionary argument keyed off integers for each angle {0: [<et>, <ec>, <es>], ...}
        self.e_allow = {'et0': et0, 'et90': et90, 'et45': et45, 'etn45': etn45, 'ec0': ec0, 'ec90': ec90,
                        'ec45': ec45, 'ecn45': ecn45, 'es0': es0, 'es90': es90, 'es45': es45, 'esn45': esn45}

    @staticmethod
    def _strain_margins(strains, et0=None, ec0=None, et90=None, ec90=None, es0=None, es90=None):
        r"""Calculates margins of safety

        Parameters
        ----------
        strains : ndarray
            2D nx3 array of [[:math: `\epsilon_x, \epsilon_y, \gamma_{xy}`], ...] in-plane strains
        et0 : float, optional
            tension strain allowable in 0 deg direction
        et90 : float, optional
            tension strain allowable in 90 deg direction
        ec0 : float, optional
            compression strain allowable in 0 deg direction
        ec90 : float, optional
            compression strain allowable in 90 deg direction
        es0 : float, optional
            shear strain allowable in 0 deg direction
        es90 : float, optional
            shear strain allowable in 90 deg direction

        Returns
        -------
        margins : ndarray
            2D nx3 array [[x-dir margin, y-dir margin, xy margin], ...]

        """
        margins = np.empty((strains.shape[0], 3))
        margins[:] = np.inf
        with np.errstate(divide='ignore'):
            # 0 deg direction
            if et0 and ec0:
                x_strains = strains[:, 0]
                margins[:, 0] = np.select(
                    [x_strains > 0, x_strains < 0], [et0/x_strains - 1, -abs(ec0)/x_strains - 1])
            # 90 deg direction
            if et90 and ec90:
                y_strains = strains[:, 1]
                margins[:, 1] = np.select(
                    [y_strains > 0, y_strains < 0], [et90/y_strains - 1, -abs(ec90)/y_strains - 1])
            # 0/90 shear
            if es0 and es90:
                xy_strains = np.abs(strains[:, 2])
                # i_nz = np.nonzero(xy_strains)
                es = min(abs(es0), abs(es90))
                margins[:, 2] = es/xy_strains - 1
        return margins

    @staticmethod
    def _rotate_strains(strains, angle=0.):
        r"""Rotates the strain components by given angle

        The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

        Parameters
        ----------
        strains : ndarray
            2D nx3 array of [:math: `\epsilon_x, \epsilon_y, \epsilon_{xy}`] in-plane strains
        angle : float, default 0.
            angle measured counter-clockwise from positive x-axis (radians)

        Returns
        -------
        ndarray
            2D nx3 array of [:math: `\epsilon_x', \epsilon_y', \epsilon_{xy}'`] rotated stresses

        """
        c = np.cos(angle)
        s = np.sin(angle)
        rotation_matrix = np.array([
            [c**2, s**2, s*c],
            [s**2, c**2, -s*c],
            [-2*s*c, 2*s*c, c**2 - s**2]
        ])
        strains = rotation_matrix @ strains.T
        return strains.T

    def analyze(self, bearing, bypass, rc=0., num=100, w=0.):
        """ Analyze the joint for a set of loads

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)

        Returns
        -------
        ndarray
            2D numx6 array of margins of safety
            [[<0 deg margin>, <90 deg margin>, <0/90 shear margin>,
            <45 deg margin>, <-45 deg margin>, <45/-45 shear margin>], ...]

        """
        e_all = self.e_allow
        margins = np.empty((num, 6))
        # check 0/90 direction
        strains = self.strains(bearing, bypass, rc=rc, num=num, w=w)
        margins[:, :3] = self._strain_margins(strains, et0=e_all['et0'], ec0=e_all['ec0'], et90=e_all['et90'],
                                              ec90=e_all['ec90'], es0=e_all['es0'], es90=e_all['es90'])
        # check 45/-45 direction
        strains = self._rotate_strains(strains, angle=np.deg2rad(45))
        margins[:, 3:] = self._strain_margins(strains, et0=e_all['et45'], ec0=e_all['ec45'], et90=e_all['etn45'],
                                              ec90=e_all['ecn45'], es0=e_all['es45'], es90=e_all['esn45'])
        return margins


