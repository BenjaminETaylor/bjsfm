import numpy as np
import matplotlib.pyplot as plt


def plot_stress(lekhnitskii_obj, comp=0, rnum=100, tnum=100, xbounds=None, ybounds=None):
    """ Plots stresses

    Parameters
    ----------
    lekhnitskii_obj : :obj: lekhnitskii.UnloadedHole or :obj: lekhnitskii.LoadedHole
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

    """
    radius = lekhnitskii_obj.r

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

    stress = lekhnitskii_obj.stress(x.flatten(), y.flatten())[:, comp]
    stress.shape = (len(x), len(y))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cp = plt.contourf(x, y, stress, corner_mask=True)
    plt.colorbar(cp)
    plt.xlim(xbounds[0], xbounds[1])
    plt.ylim(ybounds[0], ybounds[1])
    plt.title(f'Python bjsfm Stress:\n {comp} dir stress')
    plt.show()








