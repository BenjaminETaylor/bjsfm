import numpy as np
import matplotlib.pyplot as plt


def plot_stress(lekhnitskii_obj, comp=0, xnum=100, ynum=100, xbounds=None, ybounds=None):
    """ Plots stresses

    Parameters
    ----------
    lekhnitskii_obj : :obj: lekhnitskii.UnloadedHole or :obj: lekhnitskii.LoadedHole
        LoadedHole or UnloadedHole instance
    comp : {0, 1, 2}, optional
        stress component, default=0
    xnum : int, optional
        number of points to plot along x-axis, default=100
    ynum : int, optional
        number of points to plot along y-axis, default=100
    xbounds : tuple of int, optional
        (x0, x1) x-axis bounds
    ybounds : tuple of int, optional
        (y0, y1) y-axis bounds

    """
    radius = lekhnitskii_obj.r

    # set bounds
    if xbounds:
        x = np.linspace(xbounds[0], xbounds[1], endpoint=True, num=xnum)
    else:
        x = np.linspace(-6*radius, 6*radius, endpoint=True, num=xnum)
    if ybounds:
        y = np.linspace(ybounds[0], ybounds[1], endpoint=True, num=ynum)
    else:
        y = np.linspace(-6*radius, 6*radius, endpoint=True, num=ynum)

    # remove points inside hole
    X, Y = np.meshgrid(x, y)

    # get stresses
    # mX = np.ma.masked_inside(X, -radius, radius)
    # mY = np.ma.masked_inside(Y, -radius, radius)
    stress = lekhnitskii_obj.stress(X.flatten(), Y.flatten())[:, comp]
    stress.shape = (len(x), len(y))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cp = plt.contourf(X, Y, stress, corner_mask=True)
    plt.colorbar(cp)
    plt.title(f'Python bjsfm Stress:\n {comp} dir stress')
    plt.show()








