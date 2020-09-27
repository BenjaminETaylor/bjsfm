import numpy as np
import matplotlib.pyplot as plt


def plot_stress(stress_obj, component=0, refinement=100, xbounds=None, ybounds=None):
    """ Plots the x-direction stresses

    Parameters
    ----------
    stress_obj
    component
    refinement
    xbounds
    ybounds

    Returns
    -------

    """
    radius = stress_obj.r

    # set bounds
    if xbounds:
        x = np.linspace(xbounds[0], xbounds[1], endpoint=True, num=refinement)
    else:
        x = np.linspace(-6*radius, 6*radius, endpoint=True, num=refinement)
    if ybounds:
        y = np.linspace(ybounds[0], ybounds[1], endpoint=True, num=refinement)
    else:
        y = np.linspace(-6*radius, 6*radius, endpoint=True, num=refinement)

    # remove points inside hole
    X, Y = np.meshgrid(x, y)

    # get stresses
    stress = stress_obj.stress(X.flatten(), Y.flatten())[:, component]
    stress.shape = (len(x), len(y))

    fig, ax = plt.subplots()
    cp = plt.contourf(X, Y, stress, corner_mask=True)
    plt.colorbar(cp)
    plt.show()






