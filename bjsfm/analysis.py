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
from nptyping import NDArray, Shape, Float, Bool
import bjsfm.lekhnitskii as lk
from bjsfm.plotting import plot_stress, plot_displacement, plot_bearing_bypass


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

    def __init__(self, a_matrix: NDArray[Shape['3, 3'], Float], thickness: float, diameter: float) -> None:
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix, dtype=float)
        self.a_inv = np.linalg.inv(self.a)

    def _loaded(self, bearing: NDArray[Shape['2'], Float]) -> lk.LoadedHole:
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

    def _unloaded(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float], w: float = 0.) \
            -> lk.UnloadedHole:
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

    def polar_points(self, rc: float = 0., num: int = 100) \
            -> tuple[NDArray[Shape['*'], Float], NDArray[Shape['*'], Float]]:
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

    def xy_points(self, rc: float = 0., num: int = 100) \
            -> tuple[NDArray[Shape['*'], Float], NDArray[Shape['*'], Float]]:
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

    def bearing_front_mask(self, bearing: NDArray[Shape['2'], Float], num: int = 100,
                           exclusion_angle: float = 0.) -> NDArray[Shape['*'], Bool]:
        """Boolean keep-mask for analysis points, excluding those in front of the bearing load

        Notes
        -----
        Points whose polar location is within +/- ``exclusion_angle`` of the bearing load direction
        (the loaded front of the hole) are excluded (``False``). Returns an all-``True`` mask when no
        exclusion applies (``exclusion_angle`` is zero/falsy or there is no bearing load).

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force); sets the front direction
        num : int, default 100
            number of points around the hole
        exclusion_angle : float, default 0.
            +/- half-angle (degrees) in front of the bearing load to exclude

        Returns
        -------
        ndarray
            1D boolean array of length ``num`` (``True`` = keep)

        """
        p, brg_theta = self.bearing_angle(np.array(bearing, dtype=float))
        if not exclusion_angle or p == 0:
            return np.ones(num, dtype=bool)
        _, theta = self.polar_points(num=num)
        # signed angular distance from the bearing direction, wrapped to [-pi, pi]
        ang_dist = np.abs(np.angle(np.exp(1j*(theta - brg_theta))))
        return ang_dist > np.deg2rad(exclusion_angle)

    @staticmethod
    def bearing_angle(bearing: NDArray[Shape['2'], Float]) -> tuple[float, float]:
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

    def stresses(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float],
                 rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[Shape['*, 3'], Float]:
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

    def strains(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float],
                rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[Shape['*, 3'], Float]:
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

    def displacements(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float],
                 rc: float = 0., num: int = 100, w: float = 0.) -> NDArray[Shape['*, 2'], Float]:
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

    def plot_stress(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float],
                    w: float = 0., comp: str = 'x', rnum: int = 100, tnum: int = 100, axes: plt.axes = None,
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

    def plot_displacement(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float],
                          w: float = 0., comp: str = 'x', rnum: int = 100, tnum: int = 100, axes: plt.axes = None,
                          xbounds: tuple[float, float] = None, ybounds: tuple[float, float] = None,
                          cmap: str = 'jet', cmin: float = None, cmax: float = None) -> None:
        """ Plots displacements

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
        comp : {'x', 'y'}, default 'x'
            displacement component (u for 'x', v for 'y')
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
        plot_displacement(lk_1=self._unloaded(bearing, bypass, w=w), lk_2=self._loaded(bearing), comp=comp, rnum=rnum,
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

    def __init__(self, a_matrix: NDArray[Shape['3, 3'], Float], thickness: float, diameter: float,
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
        Every returned dictionary contains the union of all keys found across the input dictionaries
        (so each is as long as the longest when the others are subsets). Missing entries are filled
        with np.inf. New dictionaries are returned; the original inputs are not modified.

        Parameters
        ----------
        dicts : list of dict

        Returns
        -------
        list of dictionaries that are equal in size and contain the same keys

        """
        keys = set().union(*dicts) if dicts else set()
        return [{key: d.get(key, np.inf) for key in keys} for d in dicts]

    @staticmethod
    def _strain_margins(strains: NDArray[Shape['*, 3'], Float], et: float = None, ec: float = None,
                        es: float = None) -> NDArray[Shape['*, 2'], Float]:
        r"""Calculates margins of safety

        Notes
        -----
        Assumes strains and allowables are all in same orientation (no rotations occur). Any
        combination of allowables may be supplied; a mode whose allowable is omitted (``None``) is
        skipped and reported with an infinite margin. Tension is checked against positive normal
        strains, compression against negative normal strains, and shear against the shear strains.

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
            2D nx2 array [[tens. or comp. margin, shear margin], ...]

        """
        margins = np.empty((strains.shape[0], 2))
        margins[:] = np.inf
        x_strains = strains[:, 0]
        with np.errstate(divide='ignore'):
            # tension (checked only against positive normal strains)
            if et:
                tension = x_strains > 0
                margins[tension, 0] = et/x_strains[tension] - 1
            # compression (checked only against negative normal strains)
            if ec:
                compression = x_strains < 0
                margins[compression, 0] = -abs(ec)/x_strains[compression] - 1
            # shear
            if es:
                xy_strains = np.abs(strains[:, 2])
                margins[:, 1] = abs(es)/xy_strains - 1
        return margins

    def analyze(self, bearing: NDArray[Shape['2'], Float], bypass: NDArray[Shape['3'], Float], rc: float = 0.,
                num: int = 100, w: float = 0., exclusion_angle: float = 0.) -> NDArray[Shape['*, 6'], Float]:
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
        exclusion_angle : float, default 0.
            +/- half-angle (degrees) in front of the bearing load to exclude from the checks
            (excluded points are reported with infinite margin)

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
        # exclude points in front of the bearing load (report infinite margin so they never govern)
        keep = self.bearing_front_mask(bearing, num=num, exclusion_angle=exclusion_angle)
        margins[~keep, :] = np.inf
        return margins

    def bearing_bypass_curve(self, brg_allow: float = None, npoints: int = 100, rc: float = 0., num: int = 100,
                             w: float = 0., exclusion_angle: float = 0.) \
            -> tuple[NDArray[Shape['*'], Float], NDArray[Shape['*'], Float]]:
        r"""Generate the max-strain bearing-stress vs. bypass-strain failure envelope

        Notes
        -----
        Bearing and bypass loads are applied collinearly along the x-axis. By linearity the strain at
        any location/ply is the superposition of a unit-bearing-stress field and a unit-bypass-strain
        field, so the *combined* in-plane strain is ``brg*s + byp*e`` for bearing stress ``s`` and
        bypass strain ``e``. Every check uses this combined strain (bearing and bypass strains are
        never separated), so a ply may fail under bearing strains alone.

        For every ply direction at every location the combined fiber-axis strain is checked against
        all three max-strain modes, each with its own directional allowable: tension
        (:math: `\epsilon_x \le e_t`), compression (:math: `\epsilon_x \ge -e_c`) and shear
        (:math: `|\gamma_{xy}| \le e_s`). Each criterion is linear in ``e``,

        .. math:: c\,e \le A - B\,s

        so for a fixed bearing stress the largest admissible (tension) bypass strain is
        ``e(s) = min over c > 0 of (A - B s)/c``. The envelope is swept from zero bearing up to the
        bearing stress at which a load first fails any mode with no bypass (``e = 0``), or up to
        ``brg_allow`` if that is smaller, and is closed down to the bearing-stress axis at that end.

        Only tension bypass (``e >= 0``) is applied. Bearing stress is :math: `P/(D \cdot t)` and
        bypass strain is the far-field applied strain :math: `a^{-1}_{00} N_x`.

        Parameters
        ----------
        brg_allow : float, optional
            bearing stress allowable; cuts the envelope off at this bearing stress when supplied
        npoints : int, default 100
            number of bearing-stress steps sampled along the envelope
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)
        exclusion_angle : float, default 0.
            +/- half-angle (degrees) in front of the bearing load to exclude from the checks

        Returns
        -------
        brg_stress, byp_strain : ndarray
            1D arrays describing the (closed) failure envelope

        """
        diameter = self.r*2
        t = self.t
        a00 = self.a_inv[0, 0]
        et, ec, es = self._et, self._ec, self._es

        # unit strain fields (linear superposition): per unit bearing stress and per unit bypass strain
        brg_field = self.strains([diameter*t, 0.], [0., 0., 0.], rc=rc, num=num, w=w)
        byp_field = self.strains([0., 0.], [1./a00 if a00 else 0., 0., 0.], rc=rc, num=num, w=w)

        # exclude points in front of the bearing load (applied along +x here) from the checks
        keep = self.bearing_front_mask([1., 0.], num=num, exclusion_angle=exclusion_angle)

        # Assemble every provided max-strain criterion as a line  c*e <= A - B*s  over all
        # locations/plies. c (coef) is the bypass sensitivity, B the bearing sensitivity, A the
        # directional allowable. A criterion is added only when its allowable is finite (i.e. the
        # user provided it), so any combination of tension/compression/shear allowables is supported.
        coef, A, B = [], [], []
        for angle in self.angles:
            rad = np.deg2rad(angle)
            rb = lk.rotate_strain(brg_field, angle=rad)  # combined-strain bearing part, ply axes
            rp = lk.rotate_strain(byp_field, angle=rad)  # combined-strain bypass part, ply axes
            bn, bg = rb[keep, 0], rb[keep, 2]  # normal (eps_x) and shear (gamma_xy) per unit bearing stress
            pn, pg = rp[keep, 0], rp[keep, 2]  # normal and shear per unit bypass strain
            ones = np.ones_like(bn)
            et_a, ec_a, es_a = et[angle], abs(ec[angle]), abs(es[angle])
            if np.isfinite(et_a):   # tension      eps_x <= et   -> pn*e <= et - bn*s
                coef.append(pn);  A.append(et_a*ones); B.append(bn)
            if np.isfinite(ec_a):   # compression  eps_x >= -ec  -> -pn*e <= ec + bn*s
                coef.append(-pn); A.append(ec_a*ones); B.append(-bn)
            if np.isfinite(es_a):   # shear        |gam_xy| <= es
                coef.append(pg);  A.append(es_a*ones); B.append(bg)   # gam_xy <= es  -> pg*e <= es - bg*s
                coef.append(-pg); A.append(es_a*ones); B.append(-bg)  # gam_xy >= -es -> -pg*e <= es + bg*s
        if not coef:  # no allowables provided -> no failure envelope to draw
            return np.array([]), np.array([])
        coef = np.concatenate(coef)
        A = np.concatenate(A)
        B = np.concatenate(B)

        # bearing stress at which a load first fails any mode with no bypass (rhs A - B*s reaches 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_fail = np.where(B > 0, A/B, np.inf)
        s0 = np.min(s_fail)
        s_end = min(s0, brg_allow) if brg_allow is not None else s0
        if not np.isfinite(s_end):  # nothing limits bearing and no allowable supplied
            return np.array([]), np.array([])

        # largest admissible tension bypass strain at each bearing stress: min over c > 0 of (A-B*s)/c
        up = coef > 0
        if not np.any(up):  # no provided mode is aggravated by bypass -> envelope unbounded by strain
            return np.array([]), np.array([])
        cu, Au, Bu = coef[up], A[up], B[up]
        brg_stress = np.linspace(0., s_end, num=npoints)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = (Au[None, :] - np.outer(brg_stress, Bu))/cu[None, :]
        byp_strain = np.maximum(np.min(e, axis=1), 0.)

        # close the envelope down to the bearing-stress axis at the right end (bearing cutoff/failure)
        brg_stress = np.append(brg_stress, brg_stress[-1])
        byp_strain = np.append(byp_strain, 0.)
        return brg_stress, byp_strain

    def plot_bearing_bypass(self, brg_allow: float = None, npoints: int = 100, rc: float = 0., num: int = 100,
                            w: float = 0., exclusion_angle: float = 0., axes: plt.axes = None,
                            xbounds: tuple[float, float] = None, ybounds: tuple[float, float] = None,
                            color: str = 'C0', label: str = None) -> None:
        """ Plots the max-strain bearing-stress vs. bypass-strain failure envelope

        Parameters
        ----------
        brg_allow : float, optional
            bearing stress allowable; cuts the envelope off at this bearing stress when supplied
        npoints : int, default 100
            number of bearing-stress steps sampled along the envelope
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole
        w : float, default 0.
            pitch or width in bearing load direction
            (set to 0. for infinite plate)
        exclusion_angle : float, default 0.
            +/- half-angle (degrees) in front of the bearing load to exclude from the checks
        axes : matplotlib.axes, optional
            a custom axes to plot on
        xbounds : tuple of float, optional
            (x0, x1) x-axis bounds
        ybounds : tuple of float, optional
            (y0, y1) y-axis bounds
        color : str, optional
            line color (any matplotlib color)
        label : str, optional
            legend label for the curve

        """
        brg_stress, byp_strain = self.bearing_bypass_curve(
            brg_allow=brg_allow, npoints=npoints, rc=rc, num=num, w=w, exclusion_angle=exclusion_angle)
        plot_bearing_bypass(brg_stress, byp_strain, brg_allow=brg_allow, axes=axes,
                            xbounds=xbounds, ybounds=ybounds, color=color, label=label)

















