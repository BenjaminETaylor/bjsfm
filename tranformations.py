"""This module defines functions for transforming plate parameters

References
----------
.. [1] Lekhnitskii, S., Tsai, S., & Cheron, T. (1987). *Anisotropic plates* (2nd ed.).
   New York: Gordon and Breach science.
"""
import numpy as np


def rotate_plane_stress(stresses, angle=0.):
    r"""Rotates the stress components by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    stresses : (1, 3) ndarray
        [:math: `\sigma_x, \sigma_y, \tau_{xy}`] in-plane stresses
    angle : float
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    (1, 3) ndarray
        [:math: `\sigma_x', \sigma_y', \tau_{xy}'`] rotated stresses

    """
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([
        [c**2, s**2, 2*s*c],
        [s**2, c**2, -2*s*c],
        [-s*c, s*c, c**2-s**2]
    ])
    return rotation_matrix.dot(stresses)


def rotate_material_matrix(a_inv, angle=0.):
    r"""Rotates the material compliance matrix by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Notes
    -----
    This function implements Eq. 9.6 [1]_

    Parameters
    ----------
    a_inv : ndarray
        (3, 3) inverse CLPT A-matrix
    angle : float
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    ndarray
        (3, 3) rotated compliance matrix

    """
    c = np.cos(angle)
    s = np.sin(angle)

    a11 = a_inv[0, 0]
    a12 = a_inv[0, 1]
    a16 = a_inv[0, 2]
    a22 = a_inv[1, 1]
    a26 = a_inv[1, 2]
    a66 = a_inv[2, 2]

    a11p = a11*c**4 + (2*a12 + a66)*s**2*c**2 + a22*s**4 + (a16*c**2 + a26*s**2)*np.sin(2*angle)
    a22p = a11*s**4 + (2*a12 + a66)*s**2*c**2 + a22*c**4 - (a16*s**2 + a26*c**2)*np.sin(2*angle)
    a12p = a12 + (a11 + a22 - 2*a12 - a66)*s**2*c**2 + 0.5*(a26 - a16)*np.sin(2*angle)*np.cos(2*angle)
    a66p = a66 + 4*(a11 + a22 - 2*a12 - a66)*s**2*c**2 + 2*(a26 - a16)*np.sin(2*angle)*np.cos(2*angle)
    a16p = ((a22*s**2 - a11*c**2 + 0.5*(2*a12 + a66)*np.cos(2*angle))*np.sin(2*angle)
            + a16*c**2*(c**2 - 3*s**2) + a26*s**2*(3*c**2 - s**2))
    a26p = ((a22*c**2 - a11*s**2 - 0.5*(2*a12 + a66)*np.cos(2*angle))*np.sin(2*angle)
            + a16*s**2*(3*c**2 - s**2) + a26*c**2*(c**2 - 3*s**2))

    return np.array([[a11p, a12p, a16p], [a12p, a22p, a26p], [a16p, a26p, a66p]])


def rotate_roots(mu1, mu2, angle=0.):
    """Rotates the roots of the characteristic equation

    Notes
    -----
    This function implements Eq. 10.8 [1]_

    Parameters
    ----------
    mu1 : complex
        first root
    mu2 : complex
        second root
    angle :float
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    mu1p, mu2p : complex
        rotated roots of the characteristic equation

    """
    c = np.cos(angle)
    s = np.sin(angle)

    mu1p = (mu1*c - s)/(c + mu1*s)
    mu2p = (mu2*c - s)/(c + mu2*s)

    return mu1p, mu2p



