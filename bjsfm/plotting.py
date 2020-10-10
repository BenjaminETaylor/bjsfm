import numpy as np
import matplotlib.pyplot as plt


def plot_stress(lek_1, lek_2=None, comp=0, rnum=100, tnum=100,
                xbounds=None, ybounds=None, cmap='jet', cmin=None, cmax=None):
    """ Plots stresses

    Notes
    -----
    colormap options can be found at
        https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    Parameters
    ----------
    lek_1 : :obj: lekhnitskii.UnloadedHole or :obj: lekhnitskii.LoadedHole
        LoadedHole or UnloadedHole instance
    lek_2 : :obj: lekhnitskii.UnloadedHole or :obj: lekhnitskii.LoadedHole, optional
        LoadedHole or UnloadedHole instance
    comp : {0, 1, 2}, optional
        stress component, default=0
    rnum : int, optional
        number of points to plot along radius, default=100
    tnum : int, optional
        number of points to plot along circumference, default=100
    xbounds : tuple of int, optional
        (x0, x1) x-axis bounds, default=6*radius
    ybounds : tuple of int, optional
        (y0, y1) y-axis bounds default=6*radius
    cmap : :obj: matplotlib.colors.ColorMap, optional
        any colormap name from matplotlib.pyplot
    cmin : float, optional
        minimum value for colormap
    cmax : float, optional
        maximum value for colormap

    """
    radius = lek_1.r

    xbounds = xbounds if xbounds else [-6*radius, 6*radius]
    ybounds = ybounds if ybounds else [-6*radius, 6*radius]
    max_bounds = max(np.max(np.abs(xbounds)), np.max(np.abs(ybounds)))

    thetas = []
    radiis = []
    for step in np.linspace(0, 2*max_bounds, num=tnum, endpoint=True):
        thetas.extend(np.linspace(0, 2*np.pi, num=rnum))
        radiis.extend([radius+step]*rnum)
    x = np.array(radiis) * np.cos(thetas)
    y = np.array(radiis) * np.sin(thetas)
    x.shape = y.shape = (tnum, rnum)

    stress = lek_1.stress(x.flatten(), y.flatten())[:, comp]
    if lek_2:
        assert lek_1.r == lek_2.r, "Cannot plot plates with different radii."
        stress_2 = lek_2.stress(x.flatten(), y.flatten())[:, comp]
        stress += stress_2

    stress.shape = (tnum, rnum)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # color limits
    cmin = cmin if cmin else np.min(stress)
    cmax = cmax if cmax else np.max(stress)
    levels = np.linspace(cmin, cmax, num=11, endpoint=True)
    cp = plt.contourf(x, y, stress, levels, corner_mask=True, cmap=plt.get_cmap(cmap), extend='both')
    plt.colorbar(cp)
    # graph limits
    plt.xlim(xbounds[0], xbounds[1])
    plt.ylim(ybounds[0], ybounds[1])
    plt.title(f'Python bjsfm Stress:\n {comp} dir stress')
    plt.show()





