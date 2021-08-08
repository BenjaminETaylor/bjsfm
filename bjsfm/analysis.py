"""This module defines analysis routines and associated machinery

Notes
-----
Without laminate details like layup and ply properties, this module is limited to max strain analysis. However, with
a third party CLPT package , the parent class `Analysis` can be used to extend this module to check max stress, Tsai-Wu,
Tsai-Hill, etc.

"""
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray
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

    def __init__(self, a_matrix: NDArray[(3, 3), float], thickness: float, diameter: float) -> None:
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix, dtype=float)
        self.a_inv = np.linalg.inv(self.a)

    def _loaded(self, bearing: NDArray[2, float]) -> lk.LoadedHole:
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

    def _unloaded(self, bearing: NDArray[2, float], bypass: NDArray[3, float], w: float = 0.) -> lk.UnloadedHole:
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
            brg_dir_bypass = lk.rotate_stress(bypass, angle=theta)
            sign = np.sign(brg_dir_bypass[0]) if abs(brg_dir_bypass[0]) > 0 else 1.
            bypass += lk.rotate_stress(np.array([p/(2*w)*sign, 0., 0.]), angle=-theta)
        return lk.UnloadedHole(bypass, d, t, a_inv)

    def polar_points(self, rc: float = 0., num: int = 100) -> tuple[NDArray[Any, float], NDArray[Any, float]]:
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

    def xy_points(self, rc: float = 0., num: int = 100) -> tuple[NDArray[Any, float], NDArray[Any, float]]:
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
    def bearing_angle(bearing: NDArray[2, float]) -> tuple[float, float]:
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

    def stresses(self, bearing: NDArray[2, float], bypass: NDArray[3, float],
                 rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[(Any, 3), float]:
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
            2D numx3 array of plate stresses (sx, sy, sxy)

        """
        x, y = self.xy_points(rc=rc, num=num)
        byp = self._unloaded(bearing, bypass, w=w)
        brg = self._loaded(bearing)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        return byp_stress + brg_stress

    def strains(self, bearing: NDArray[2, float], bypass: NDArray[3, float],
                rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[(Any, 3), float]:
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

    def displacements(self, bearing: NDArray[2, float], bypass: NDArray[3, float],
                 rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[(Any, 3), float]:
        """ Calculate displacements

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
            2D numx2 array of plate displacements (u, v)

        """
        x, y = self.xy_points(rc=rc, num=num)
        byp = self._unloaded(bearing, bypass, w=w)
        brg = self._loaded(bearing)
        byp_displacement = byp.displacement(x, y)
        brg_displacement = brg.displacement(x, y)
        return byp_displacement + brg_displacement

    def plot_stress(self, bearing: NDArray[2, float], bypass: NDArray[3, float], w: float = 0., comp: str = 'x',
                    rnum: int = 100, tnum: int = 100, axes: plt.axes = None,
                    xbounds: tuple[float, float] = None, ybounds: tuple[float, float] = None,
                    cmap: str = 'jet', cmin: float = None, cmax: float = None) -> None:
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
        comp : {'x', 'y', 'xy'}, default 'x'
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
    et : dict, optional
        tension strain allowables (one of `et`, `ec` or `es` must be specified to obtain results)
        {<angle>: <value>, ...}
    ec : dict, optional
        compression strain allowables (one of `et`, `ec` or `es` must be specified to obtain results)
        {<angle>: <value>, ...}
    es : dict, optional
        shear strain allowables (one of `et`, `ec` or `es` must be specified to obtain results)
        {<angle>: <value>, ...}

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
    angles : list
        analysis angles (sorted)
    et : dict
        tension strain allowables
    ec : dict
        compression strain allowables
    es : dict
        shear strain allowables

    """

    def __init__(self, a_matrix: NDArray[(3, 3), float], thickness: float, diameter: float,
                 et: dict[int, float] = {}, ec: dict[int, float] = {}, es: dict[int, float] = {}) -> None:
        super(MaxStrain, self).__init__(a_matrix, thickness, diameter)
        self._et, self._ec, self._es = self._equalize_dicts([et, ec, es])
    
    @property
    def et(self) -> dict[int, float]:
        return self._et
    
    @property
    def ec(self) -> dict[int, float]:
        return self._ec
    
    @property
    def es(self) -> dict[int, float]:
        return self._es
    
    @property
    def angles(self) -> list[int]:
        return sorted(self._et.keys())

    @staticmethod
    def _equalize_dicts(dicts: list[dict]) -> list[dict]:
        """This method makes sure all dictionaries are the same size and contain the same keys

        Notes
        -----
        Maintains original contents of each dictionary, fills empty slots with np.inf, does
        not guarantee the same insertion order in each dictionary

        Parameters
        ----------
        dicts : list of dict

        Returns
        -------
        list of dictionaries that are equal in size and contain the same keys (without modifying original contents)

        """
        for d in dicts:
            others = dicts.copy()
            others.remove(d)
            for od in others:
                for key in d:
                    if key not in od:
                        od[key] = np.inf
        return dicts

    @staticmethod
    def _strain_margins(strains: NDArray[(Any, 3), float], et: float = None, ec: float = None,
                        es: float = None) -> NDArray[(Any, 2), float]:
        r"""Calculates margins of safety

        Notes
        -----
        Assumes strains and allowables are all in same orientation (no rotations occur)

        Parameters
        ----------
        strains : ndarray
            2D nx3 array of [[:math: `\epsilon_x, \epsilon_y, \gamma_{xy}`], ...] in-plane strains
        et : float, optional
            tension strain allowable
        ec : float, optional
            compression strain allowable
        es : float, optional
            shear strain allowable

        Returns
        -------
        margins : ndarray
            2D nx2 array [[tens./comp. margin, shear margin], ...]

        """
        margins = np.empty((strains.shape[0], 2))
        margins[:] = np.inf
        with np.errstate(divide='ignore'):
            # tension/compression
            if et and ec:
                x_strains = strains[:, 0]
                margins[:, 0] = np.select(
                    [x_strains > 0, x_strains < 0], [et/x_strains - 1, -abs(ec)/x_strains - 1])
            # shear
            if es:
                xy_strains = np.abs(strains[:, 2])
                # i_nz = np.nonzero(xy_strains)
                es = abs(es)
                margins[:, 1] = es/xy_strains - 1
        return margins

    def analyze(self, bearing: NDArray[2, float], bypass: NDArray[3, float], rc: float = 0.,
                num: int = 100, w: float = 0.) -> NDArray[(Any, 6), float]:
        """Analyze the joint for a set of loads

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
            2D [num x <number of angles>*2] array of margins of safety

        """
        et, ec, es = self._et, self._ec, self._es
        num_angles = len(self._et)
        margins = np.empty((num, 2*num_angles))
        strains = self.strains(bearing, bypass, rc=rc, num=num, w=w)
        for iangle, angle in enumerate(self.angles):  # et, es and ec are forced to have same keys in constructor
            idx = iangle*2
            allowables = {'et': et[angle], 'ec': ec[angle], 'es': es[angle]}
            rotated_strains = lk.rotate_strain(strains, angle=np.deg2rad(angle))
            margins[:, idx:idx+2] = self._strain_margins(rotated_strains, **allowables)
        return margins

















