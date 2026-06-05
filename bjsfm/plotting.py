from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from bjsfm._typing import FloatArray
from bjsfm.lekhnitskii import LoadedHole, UnloadedHole


HoleObject = Union[LoadedHole, UnloadedHole]
Component = Union['x', 'y', 'xy']
DisplacementComponent = Union['x', 'y']


def plot_stress(lk_1: HoleObject, lk_2: HoleObject = None, comp: Component = 'x', rnum: int = 100, tnum: int = 100,
                axes: Optional[Axes] = None, xbounds: Optional[tuple[float, float]] = None,
                ybounds: Optional[tuple[float, float]] = None,
                cmap: str ='jet', cmin: Optional[float] = None, cmax: Optional[float] = None) -> None:
    """ Plots stresses

    Notes
    -----
    colormap options can be found at
        https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    Parameters
    ----------
    lk_1 : bjsfm.lekhnitskii.UnloadedHole or bjsfm.lekhnitskii.LoadedHole
        LoadedHole or UnloadedHole instance
    lk_2 : bjsfm.lekhnitskii.UnloadedHole or bjsfm.lekhnitskii.LoadedHole, optional
        LoadedHole or UnloadedHole instance
    comp : {'x', 'y', 'xy'}, default 'x'
        stress component
    rnum : int, default 100
        number of points to plot along radius
    tnum : int, default 100
        number of points to plot along circumference
    axes : matplotlib.axes, optional
        a custom axes to plot on
    xbounds : tuple of float, optional
        (x0, x1) x-axis bounds, default=6*radius
    ybounds : tuple of float, optional
        (y0, y1) y-axis bounds default=6*radius
    cmap : str, optional
        name of any colormap name from matplotlib.pyplot
    cmin : float, optional
        minimum value for colormap
    cmax : float, optional
        maximum value for colormap

    """
    convert_comp = {'x': 0, 'y': 1, 'xy': 2}
    radius = lk_1.r

    xbounds = xbounds if xbounds else [-6*radius, 6*radius]
    ybounds = ybounds if ybounds else [-6*radius, 6*radius]
    max_bounds = max(np.max(np.abs(xbounds)), np.max(np.abs(ybounds)))

    thetas = []
    radii = []
    for step in np.linspace(0, 2*max_bounds, num=tnum, endpoint=True):
        thetas.extend(np.linspace(0, 2*np.pi, num=rnum))
        radii.extend([radius+step]*rnum)
    x = np.array(radii) * np.cos(thetas)
    y = np.array(radii) * np.sin(thetas)
    x.shape = y.shape = (tnum, rnum)

    stress = lk_1.stress(x.flatten(), y.flatten())[:, convert_comp[comp]]
    if lk_2:
        assert lk_1.r == lk_2.r, "Cannot plot plates with different hole diameters."
        stress_2 = lk_2.stress(x.flatten(), y.flatten())[:, convert_comp[comp]]
        stress += stress_2

    stress.shape = (tnum, rnum)

    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
        fig = ax.figure
    ax.set_aspect('equal')
    # color limits
    cmin = cmin if cmin else np.min(stress)
    cmax = cmax if cmax else np.max(stress)
    levels = np.linspace(cmin, cmax, num=11, endpoint=True)
    cp = ax.contourf(x, y, stress, levels, corner_mask=True, cmap=plt.get_cmap(cmap), extend='both')
    fig.colorbar(cp, ax=ax)
    # graph limits
    ax.set_xlim(xbounds[0], xbounds[1])
    ax.set_ylim(ybounds[0], ybounds[1])
    ax.set_title(f'Python bjsfm Stress:\n {comp} dir stress')
    if not axes:
        plt.show()


def plot_displacement(lk_1: HoleObject, lk_2: HoleObject = None, comp: DisplacementComponent = 'x', rnum: int = 100,
                      tnum: int = 100, axes: Optional[Axes] = None, xbounds: Optional[tuple[float, float]] = None,
                      ybounds: Optional[tuple[float, float]] = None, cmap: str = 'jet', cmin: Optional[float] = None,
                      cmax: Optional[float] = None) -> None:
    """ Plots displacements

    Notes
    -----
    colormap options can be found at
        https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    Parameters
    ----------
    lk_1 : bjsfm.lekhnitskii.UnloadedHole or bjsfm.lekhnitskii.LoadedHole
        LoadedHole or UnloadedHole instance
    lk_2 : bjsfm.lekhnitskii.UnloadedHole or bjsfm.lekhnitskii.LoadedHole, optional
        LoadedHole or UnloadedHole instance
    comp : {'x', 'y'}, default 'x'
        displacement component (u for 'x', v for 'y')
    rnum : int, default 100
        number of points to plot along radius
    tnum : int, default 100
        number of points to plot along circumference
    axes : matplotlib.axes, optional
        a custom axes to plot on
    xbounds : tuple of float, optional
        (x0, x1) x-axis bounds, default=6*radius
    ybounds : tuple of float, optional
        (y0, y1) y-axis bounds default=6*radius
    cmap : str, optional
        name of any colormap name from matplotlib.pyplot
    cmin : float, optional
        minimum value for colormap
    cmax : float, optional
        maximum value for colormap

    """
    convert_comp = {'x': 0, 'y': 1}
    radius = lk_1.r

    xbounds = xbounds if xbounds else [-6*radius, 6*radius]
    ybounds = ybounds if ybounds else [-6*radius, 6*radius]
    max_bounds = max(np.max(np.abs(xbounds)), np.max(np.abs(ybounds)))

    thetas = []
    radii = []
    for step in np.linspace(0, 2*max_bounds, num=tnum, endpoint=True):
        thetas.extend(np.linspace(0, 2*np.pi, num=rnum))
        radii.extend([radius+step]*rnum)
    x = np.array(radii) * np.cos(thetas)
    y = np.array(radii) * np.sin(thetas)
    x.shape = y.shape = (tnum, rnum)

    displacement = lk_1.displacement(x.flatten(), y.flatten())[:, convert_comp[comp]]
    if lk_2:
        assert lk_1.r == lk_2.r, "Cannot plot plates with different hole diameters."
        displacement_2 = lk_2.displacement(x.flatten(), y.flatten())[:, convert_comp[comp]]
        displacement += displacement_2

    displacement.shape = (tnum, rnum)

    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
        fig = ax.figure
    ax.set_aspect('equal')
    # color limits
    cmin = cmin if cmin else np.min(displacement)
    cmax = cmax if cmax else np.max(displacement)
    levels = np.linspace(cmin, cmax, num=11, endpoint=True)
    cp = ax.contourf(x, y, displacement, levels, corner_mask=True, cmap=plt.get_cmap(cmap), extend='both')
    fig.colorbar(cp, ax=ax)
    # graph limits
    ax.set_xlim(xbounds[0], xbounds[1])
    ax.set_ylim(ybounds[0], ybounds[1])
    ax.set_title(f'Python bjsfm Displacement:\n {comp} dir displacement')
    if not axes:
        plt.show()


def plot_bearing_bypass(brg_stress: FloatArray, byp_strain: FloatArray,
                        brg_allow: Optional[float] = None, axes: Optional[Axes] = None,
                        xbounds: Optional[tuple[float, float]] = None,
                        ybounds: Optional[tuple[float, float]] = None, color: str = 'C0',
                        label: Optional[str] = None) -> None:
    """ Plots the max-strain bearing-stress vs. bypass-strain failure envelope

    Notes
    -----
    The (bearing stress, bypass strain) points describe a closed max-strain failure envelope built
    from combined bearing + bypass strains (tension, compression and shear). When ``brg_allow`` is
    supplied a dashed vertical line marks the bearing-stress cutoff.

    Parameters
    ----------
    brg_stress : array_like
        1D array of bearing stresses (x-axis), e.g. from ``MaxStrain.bearing_bypass_curve``
    byp_strain : array_like
        1D array of bypass strains (y-axis), e.g. from ``MaxStrain.bearing_bypass_curve``
    brg_allow : float, optional
        bearing stress allowable; drawn as a dashed vertical cutoff line when supplied
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
    brg_stress = np.asarray(brg_stress, dtype=float)
    byp_strain = np.asarray(byp_strain, dtype=float)

    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
        fig = ax.figure

    ax.plot(brg_stress, byp_strain, color=color, label=label)
    if brg_allow is not None:
        ax.axvline(brg_allow, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlim(xbounds if xbounds else (0., None))
    ax.set_ylim(ybounds if ybounds else (0., None))
    ax.set_xlabel('Bearing Stress')
    ax.set_ylabel('Bypass Strain')
    ax.set_title('Python bjsfm Bearing/Bypass Diagram')
    if label:
        ax.legend()
    if not axes:
        plt.show()





