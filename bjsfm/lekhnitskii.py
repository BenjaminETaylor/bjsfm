"""Lekhnitskii solutions to homogeneous anisotropic plates with loaded and unloaded holes

Notes
-----
This module uses the following acronyms
* CLPT: Classical Laminated Plate Theory

References
----------
.. [1] Esp, B. (2007). *Stress distribution and strength prediction of composite
   laminates with multiple holes* (PhD thesis). Retrieved from
   https://rc.library.uta.edu/uta-ir/bitstream/handle/10106/767/umi-uta-1969.pdf?sequence=1&isAllowed=y
.. [2] Lekhnitskii, S., Tsai, S., & Cheron, T. (1987). *Anisotropic plates* (2nd ed.).
   New York: Gordon and Breach science.
.. [3] Garbo, S. and Ogonowski, J. (1981) *Effect of variances and manufacturing
   tolerances on the design strength and life of mechanically fastened
   composite joints* (Vol. 1,2,3). AFWAL-TR-81-3041.
.. [4] Waszczak, J.P. and Cruse T.A. (1973) *A synthesis procedure for mechanically
   fastened joints in advanced composite materials* (Vol. II). AFML-TR-73-145.

"""
import logging
import abc
from collections.abc import Callable
from typing import Any
import numpy as np
import numpy.testing as nptest
from nptyping import NDArray


logger = logging.getLogger(__name__)


def rotate_stress(stresses: NDArray[3, np.float], angle: float = 0.) -> NDArray[3, np.float]:
    r"""Rotates 2D stress components by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    stresses : ndarray
        array of [:math: `\sigma_x, \sigma_y, \tau_{xy}`] in-plane stresses
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    ndarray
        2D array of [:math: `\sigma_x', \sigma_y', \tau_{xy}'`] rotated stresses

    """
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([
        [c**2, s**2, 2*s*c],
        [s**2, c**2, -2*s*c],
        [-s*c, s*c, c**2-s**2]
    ])
    stresses = rotation_matrix @ stresses.T
    return stresses.T


def rotate_strain(strains: NDArray[3, np.float], angle: float = 0.) -> NDArray[3, float]:
    r"""Rotates 2D strain components by given angle

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


def rotate_material_matrix(a_inv: NDArray[(3, 3), np.float], angle: float = 0.) -> NDArray[(3, 3), float]:
    r"""Rotates the material compliance matrix by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Notes
    -----
    This function implements Eq. 9.6 [1]_

    Parameters
    ----------
    a_inv : ndarray
        2D (3, 3) inverse CLPT A-matrix
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    ndarray
        2D (3, 3) rotated compliance matrix

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

    # test invariants (Eq. 9.7 [2]_)
    nptest.assert_almost_equal(a11p + a22p + 2*a12p, a11 + a22 + 2*a12, decimal=4)
    nptest.assert_almost_equal(a66p - 4*a12p, a66 - 4*a12, decimal=4)

    return np.array([[a11p, a12p, a16p], [a12p, a22p, a26p], [a16p, a26p, a66p]])


def rotate_complex_parameters(mu1: complex, mu2: complex, angle: float = 0.) -> tuple[complex, complex]:
    r"""Rotates the complex parameters by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Notes
    -----
    Implements Eq. 10.8 [2]_

    Parameters
    ----------
    mu1 : complex
        first complex parameter
    mu2 : complex
        second complex parameter
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    mu1p, mu2p : complex
        first and second transformed complex parameters

    """
    c = np.cos(angle)
    s = np.sin(angle)

    mu1p = (mu1*c - s)/(c + mu1*s)
    mu2p = (mu2*c - s)/(c + mu2*s)

    return mu1p, mu2p


class Hole(abc.ABC):
    """Abstract parent class for defining a hole in an anisotropic infinite plate

    This class defines shared methods and attributes for anisotropic elasticity solutions of plates with circular
    holes.

    This is an abstract class, do not instantiate this class.

    Notes
    -----
    The following assumptions apply for plates in a state of generalized plane stress.

    #. The plates are homogeneous and a plane of elastic symmetry which is parallel to their middle plane
       exists at every point.
    #. Applied forces act within planes that are parallel and symmetric to the middle plane of the plates,
       and have negligible variation through the thickness.
    #. Plate deformations are small.

    Parameters
    ----------
    diameter : float
        hole diameter
    thickness : float
        laminate thickness
    a_inv : array_like
        2D (3, 3) inverse of CLPT A-matrix

    Attributes
    ----------
    r : float
        the hole radius
    a : ndarray
        (3, 3) inverse a-matrix of the laminate
    h : float
        thickness of the laminate
    mu1 : float
        real part of first root of characteristic equation
    mu2 : float
        real part of second root of characteristic equation
    mu1_bar : float
        imaginary part of first root of characteristic equation
    mu2_bar : float
        imaginary part of second root of characteristic equation

    """

    MAPPING_PRECISION = 0.0000001

    def __init__(self, diameter: float, thickness: float, a_inv: NDArray[(3, 3), float]) -> None:
        self.r = diameter/2.
        self.a = np.array(a_inv, dtype=float)
        self.h = thickness
        self.mu1, self.mu2, self.mu1_bar, self.mu2_bar = self.roots()

    def roots(self) -> tuple[complex, complex, complex, complex]:
        r""" Finds the roots to the characteristic equation

        Notes
        -----
        This method implements Eq. A.2 [1]_ or Eq. 7.4 [2]_

        .. math:: a_11\mu^4-2a_16\mu^3+(2a_12+a_66)\mu^2-2a_26\mu+a_22=0

        Raises
        ------
        ValueError
            If roots cannot be found

        """
        a11 = self.a[0, 0]
        a12 = self.a[0, 1]
        a16 = self.a[0, 2]
        a22 = self.a[1, 1]
        a26 = self.a[1, 2]
        a66 = self.a[2, 2]

        roots = np.roots([a11, -2 * a16, (2 * a12 + a66), -2 * a26, a22])

        if np.imag(roots[0]) >= 0.0:
            mu2 = roots[0]
            mu2_bar = roots[1]
        elif np.imag(roots[1]) >= 0.0:
            mu2 = roots[1]
            mu2_bar = roots[0]
        else:
            raise ValueError("mu1 cannot be solved")

        if np.imag(roots[2]) >= 0.0:
            mu1 = roots[2]
            mu1_bar = roots[3]
        elif np.imag(roots[3]) >= 0.0:
            mu1 = roots[3]
            mu1_bar = roots[2]
        else:
            raise ValueError("mu2 cannot be solved")

        return mu1, mu2, mu1_bar, mu2_bar

    def xi_1(self, z1s: NDArray[Any, complex]) -> tuple[NDArray[Any, complex], NDArray[Any, int]]:
        r"""Calculates the first mapping parameters

        Notes
        -----
        This method implements Eq. A.4 & Eq. A.5, [1]_ or Eq. 37.4 [2]_

        .. math:: \xi_1=\frac{z_1\pm\sqrt{z_1^2-a^2-\mu_1^2b^2}}{a-i\mu_1b}

        Parameters
        ----------
        z1s : ndarray
            1D array of first parameters from the complex plane :math: `z_1=x+\mu_1y`

        Returns
        -------
        xi_1s : ndarray
            1D array of the first mapping parameters
        sign_1s : ndarray
            1D array of signs producing positive mapping parameters

        """
        mu1 = self.mu1
        a = self.r
        b = self.r

        xi_1s = np.zeros(len(z1s), dtype=complex)
        sign_1s = np.zeros(len(z1s), dtype=int)

        xi_1_pos = (z1s + np.sqrt(z1s * z1s - a * a - mu1 * mu1 * b * b)) / (a - 1j * mu1 * b)
        xi_1_neg = (z1s - np.sqrt(z1s * z1s - a * a - mu1 * mu1 * b * b)) / (a - 1j * mu1 * b)

        pos_indices = np.where(np.abs(xi_1_pos) >= (1. - self.MAPPING_PRECISION))[0]
        neg_indices = np.where(np.abs(xi_1_neg) >= (1. - self.MAPPING_PRECISION))[0]

        xi_1s[pos_indices] = xi_1_pos[pos_indices]
        xi_1s[neg_indices] = xi_1_neg[neg_indices]

        # high level check that all indices were mapped
        if not (pos_indices.size + neg_indices.size) == xi_1s.size:
            bad_indices = np.where(xi_1s == 0)[0]
            logger.warning(f"xi_1 unsolvable\n Failed Indices: {bad_indices}")

        sign_1s[pos_indices] = 1
        sign_1s[neg_indices] = -1

        return xi_1s, sign_1s

    def xi_2(self, z2s: NDArray[Any, complex]) -> tuple[NDArray[Any, complex], NDArray[Any, int]]:
        r""" Calculates the first mapping parameters

        Notes
        -----
        This method implements Eq. A.4 & Eq. A.5, [1]_ or Eq. 37.4 [2]_

        .. math:: \xi_2=\frac{z_2\pm\sqrt{z_2^2-a^2-\mu_2^2b^2}}{a-i\mu_2b}

        Parameters
        ----------
        z2s : ndarray
            1D array of first parameters from the complex plane :math: `z_1=x+\mu_1y`

        Returns
        -------
        xi_2s : ndarray
            1D array of the first mapping parameters
        sign_2s : ndarray
            1D array of signs producing positive mapping parameters

        """
        mu2 = self.mu2
        a = self.r
        b = self.r

        xi_2s = np.zeros(len(z2s), dtype=complex)
        sign_2s = np.zeros(len(z2s), dtype=int)

        xi_2_pos = (z2s + np.sqrt(z2s * z2s - a * a - mu2 * mu2 * b * b)) / (a - 1j * mu2 * b)
        xi_2_neg = (z2s - np.sqrt(z2s * z2s - a * a - mu2 * mu2 * b * b)) / (a - 1j * mu2 * b)

        pos_indices = np.where(np.abs(xi_2_pos) >= (1. - self.MAPPING_PRECISION))[0]
        neg_indices = np.where(np.abs(xi_2_neg) >= (1. - self.MAPPING_PRECISION))[0]

        xi_2s[pos_indices] = xi_2_pos[pos_indices]
        xi_2s[neg_indices] = xi_2_neg[neg_indices]

        # high level check that all indices were mapped
        if not (pos_indices.size + neg_indices.size) == xi_2s.size:
            bad_indices = np.where(xi_2s == 0)[0]
            logger.warning(f"xi_2 unsolvable\n Failed Indices: {bad_indices}")

        sign_2s[pos_indices] = 1
        sign_2s[neg_indices] = -1

        return xi_2s, sign_2s

    @abc.abstractmethod
    def phi_1(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        raise NotImplementedError("You must implement this function.")

    @abc.abstractmethod
    def phi_2(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        raise NotImplementedError("You must implement this function.")

    @abc.abstractmethod
    def phi_1_prime(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        raise NotImplementedError("You must implement this function.")

    @abc.abstractmethod
    def phi_2_prime(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        raise NotImplementedError("You must implement this function.")

    def stress(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 3), float]:
        r""" Calculates the stress at (x, y) points in the plate

        Notes
        -----
        This method implements Eq. 8.2 [2]_

        .. math:: \sigma_x=2Re[\mu_1^2\Phi_1'(z_1)+\mu_2^2\Phi_2'(z_2)]
        .. math:: \sigma_y=2Re[\Phi_1'(z_1)+\Phi_2'(z_2)]
        .. math:: \tau_xy=-2Re[\mu_1\Phi_1'(z_1)+\mu_2\Phi_2'(z_2)]

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        ndarray
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
            (n, 3) in-plane stress components in the cartesian coordinate system

        """
        mu1 = self.mu1
        mu2 = self.mu2

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1_prime = self.phi_1_prime(z1)
        phi_2_prime = self.phi_2_prime(z2)

        sx = 2.0 * np.real(mu1 * mu1 * phi_1_prime + mu2 * mu2 * phi_2_prime)
        sy = 2.0 * np.real(phi_1_prime + phi_2_prime)
        sxy = -2.0 * np.real(mu1 * phi_1_prime + mu2 * phi_2_prime)

        return np.array([sx, sy, sxy]).T

    def displacement(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 2), float]:
        r""" Calculates the displacement at (x, y) points in the plate

        Notes
        -----
        This method implements Eq. 8.3 [2]_

        .. math:: u=2Re[p_1\Phi_1(z_1)+p_2\Phi_2(z_2)]
        .. math:: v=2Re[q_1\Phi_1(z_1)+q_2\Phi_2(z_2)]

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        ndarray
            [[u0, v0], [u1, v1], ... , [un, vn]]
            (n, 2) in-plane displacement components in the cartesian coordinate system

        """
        a11 = self.a[0, 0]
        a12 = self.a[0, 1]
        a16 = self.a[0, 2]
        a22 = self.a[1, 1]
        a26 = self.a[1, 2]
        mu1 = self.mu1
        mu2 = self.mu2

        p1 = a11*mu1**2 + a12 - a16*mu1
        p2 = a11*mu2**2 + a12 - a16*mu2
        q1 = a12*mu1 + a22/mu1 - a26
        q2 = a12*mu2 + a22/mu2 - a26

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1 = self.phi_1(z1)
        phi_2 = self.phi_2(z2)

        u = 2.0 * np.real(p1 * phi_1 + p2 * phi_2)
        v = 2.0 * np.real(q1 * phi_1 + q2 * phi_2)

        return np.array([u, v]).T


class UnloadedHole(Hole):
    r"""Class for defining an unloaded hole in an infinite anisotropic homogeneous plate

    This class represents an infinite anisotropic plate with a unfilled circular hole loaded at infinity with
    forces in the x, y and xy (shear) directions.

    Parameters
    ----------
    loads: array_like
        1D array [Nx, Ny, Nxy] force / unit length
    diameter: float
        hole diameter
    thickness: float
        laminate thickness
    a_inv: array_like
        2D array (3, 3) inverse CLPT A-matrix

    Attributes
    ----------
    applied_stress : (1, 3) ndarray
        [:math:`\sigma_x^*, \sigma_y^*, \tau_{xy}^*`] stresses applied at infinity

    """

    def __init__(self, loads: NDArray[3, float], diameter: float, thickness: float,
                 a_inv: NDArray[(3, 3), float]) -> None:
        super().__init__(diameter, thickness, a_inv)
        self.applied_stress = np.array(loads, dtype=float) / self.h

    def alpha(self) -> complex:
        r"""Calculates the alpha loading term for three components of applied stress at infinity

        Three components of stress are [:math:`\sigma_{x}^*, \sigma_{y}^*, \tau_{xy}^*`]

        Notes
        -----
        This method implements Eq. A.7 [1]_ which is a combination of Eq. 38.12 & Eq. 38.18 [2]_

        .. math:: \alpha_1=\frac{r}{2}(\tau_{xy}^*i-\sigma_{y}^*)

        Returns
        -------
        complex
            first fourier series term for applied stress at infinity

        """
        sy = self.applied_stress[1]
        sxy = self.applied_stress[2]
        r = self.r

        return 1j * sxy * r / 2 - sy * r / 2

    def beta(self) -> complex:
        r"""Calculates the beta loading term for three components of applied stress at infinity

        Three components of stress are [:math:`\sigma_x^*, \sigma_y^*, \tau_{xy}^*`]

        Notes
        -----
        This method implements Eq. A.7 [1]_ which is a combination of Eq. 38.12 & Eq. 38.18 [2]_

        .. math:: \beta_1=\frac{r}{2}(\tau_{xy}^*-\sigma_x^*i)

        Returns
        -------
        complex
            first fourier series term for applied stresses at infinity

        """
        sx = self.applied_stress[0]
        sxy = self.applied_stress[2]
        r = self.r

        return sxy * r / 2 - 1j * sx * r / 2

    def phi_1(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates the first stress function

        Notes
        -----
        This method implements Eq. A.6 [1]_

        .. math:: C_1=\frac{\beta_1-\mu_2\alpha_1}{\mu_1-\mu_2}
        .. math:: \Phi_1=\frac{C_1}{\xi_1}

        Parameters
        ----------
        z1 : ndarray
            1D complex array first mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        alpha = self.alpha()
        beta = self.beta()
        xi_1, sign_1 = self.xi_1(z1)

        C1 = (beta - mu2 * alpha) / (mu1 - mu2)

        return C1 / xi_1

    def phi_2(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates the second stress function

        Notes
        -----
        This method implements Eq. A.6 [1]_

        .. math:: C_2=-\frac{\beta_1-\mu_1\alpha_1}{\mu_1-\mu_2}
        .. math:: \Phi_2=\frac{C_2}{\xi_2}

        Parameters
        ----------
        z2 : ndarray
            1D complex array second mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        alpha = self.alpha()
        beta = self.beta()
        xi_2, sign_2 = self.xi_2(z2)

        C2 = -(beta - mu1 * alpha) / (mu1 - mu2)

        return C2 / xi_2

    def phi_1_prime(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates derivative of the first stress function

        Notes
        -----
        This method implements Eq. A.8 [1]_

        .. math:: C_1=\frac{\beta_1-\mu_2\alpha_1}{\mu_1-\mu_2}
        .. math:: \eta_1=\frac{z_1\pm\sqrt{z_1^2-a^2-\mu_1^2b^2}}{a-i\mu_1b}
        .. math:: \kappa_1=\frac{1}{a-i\mu_1b}
        .. math:: \Phi_1'=-\frac{C_1}{\xi_1^2}(1+\frac{z_1}{\eta_1})\kappa_1

        Parameters
        ----------
        z1 : ndarray
            1D complex array first mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        a = self.r
        b = self.r
        mu1 = self.mu1
        mu2 = self.mu2
        alpha = self.alpha()
        beta = self.beta()
        xi_1, sign_1 = self.xi_1(z1)

        C1 = (beta - mu2 * alpha) / (mu1 - mu2)
        eta1 = sign_1 * np.sqrt(z1 * z1 - a * a - mu1 * mu1 * b * b)
        kappa1 = 1 / (a - 1j * mu1 * b)

        return -C1 / (xi_1 ** 2) * (1 + z1 / eta1) * kappa1

    def phi_2_prime(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates derivative of the second stress function

        Notes
        -----
        This method implements Eq. A.8 [1]_

        .. math:: C_2=-\frac{\beta_1-\mu_1\alpha_1}{\mu_1-\mu_2}
        .. math:: \eta_2=\frac{z_2\pm\sqrt{z_2^2-a^2-\mu_2^2b^2}}{a-i\mu_2b}
        .. math:: \kappa_2=\frac{1}{a-i\mu_2b}
        .. math:: \Phi_2'=-\frac{C_2}{\xi_2^2}(1+\frac{z_2}{\eta_2})\kappa_2

        Parameters
        ----------
        z2 : ndarray
            1D complex array second mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        a = self.r
        b = self.r
        mu1 = self.mu1
        mu2 = self.mu2
        alpha = self.alpha()
        beta = self.beta()
        xi_2, sign_2 = self.xi_2(z2)

        C2 = -(beta - mu1 * alpha) / (mu1 - mu2)
        eta2 = sign_2 * np.sqrt(z2 * z2 - a * a - mu2 * mu2 * b * b)
        kappa2 = 1 / (a - 1j * mu2 * b)

        return -C2 / (xi_2 ** 2) * (1 + z2 / eta2) * kappa2

    def stress(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 3), float]:
        r""" Calculates the stress at (x, y) points in the plate

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        ndarray
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
            (n, 3) in-plane stress components in the cartesian coordinate system

        """
        sx, sy, sxy = super().stress(x, y).T

        sx_app = self.applied_stress[0]
        sy_app = self.applied_stress[1]
        sxy_app = self.applied_stress[2]

        return np.array([sx + sx_app, sy + sy_app, sxy + sxy_app]).T


def _remove_bad_displacments(displacement_func: 
    Callable[[object, NDArray[Any, float], NDArray[Any, float]], NDArray[(Any, 2), float]]): 
    """ removes displacements that are 180 degrees behind bearing load direction"""
    def inner(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 2), float]:
        # call displacement function
        displacements = displacement_func(self, x, y)
        # check if any points are 180 degrees behind bearing load
        r, angles = self._cartesian_to_polar(x, y)
        bad_angle = np.pi if self.theta == 0 else -1*(np.pi - self.theta)
        # if so, replace those results with np.nan
        displacements[np.isclose(angles, bad_angle)] = np.nan
        return displacements

    return inner 


class LoadedHole(Hole):
    """Class for defining a loaded hole in an infinite anisotropic homogeneous plate

    A cosine bearing load distribution is assumed to apply to the inside of the hole.

    Notes
    -----
    Bearing distribution as shown below Ref. [4]_

    .. image:: ../img/cosine_distribution.png
       :height: 400px

    Parameters
    ----------
    load : float
        bearing force
    diameter : float
        hole diameter
    thickness : float
        plate thickness
    a_inv : array_like
        2D array (3, 3) inverse CLPT A-matrix
    theta : float, optional
        bearing angle counter clock-wise from positive x-axis (radians)

    Attributes
    ----------
    p : float
        bearing force
    theta : float
        bearing angle counter clock-wise from positive x-axis (radians)
    A : float
        real part of equilibrium constant for first stress function
    A_bar : float
        imaginary part of equilibrium constant for first stress function
    B : float
        real part of equilibrium constant for second stress function
    B_bar : float
        imaginary part of equilibrium constant for second stress function

    """
    FOURIER_TERMS = 45  # number of fourier series terms [3]_

    def __init__(self, load: float, diameter: float, thickness: float,
                 a_inv: NDArray[(3, 3), float], theta: float = 0.) -> None:
        a_inv = rotate_material_matrix(a_inv, angle=theta)
        super().__init__(diameter, thickness, a_inv)
        self.p = load
        self.theta = theta
        self.A, self.A_bar, self.B, self.B_bar = self.equilibrium_constants()

    def alpha(self) -> NDArray[FOURIER_TERMS, complex]:
        r"""Fourier series coefficients modified for use in stress function equations

        Notes
        -----
        Exact solution:

        .. math:: \frac{P}{2\pi}\int_{-\pi/2}^{\pi/2} \cos^2 \theta \left( \cos m*\theta - i \sin m*\theta \right) \,d\theta
        .. math:: = \frac{-2 P sin(\pi m/2)}{\pi m(m^2-4)}

        Modifications to the Fourier series coefficients are developed in Eq. 37.2 [2]_

        Returns
        -------
        ndarray

        """
        h = self.h
        p = self.p
        N = self.FOURIER_TERMS
        m = np.arange(3, N + 1)

        # modification from Eq. 37.2 [2]_
        mod = -1/(h*np.pi)

        alpha = np.zeros(N)
        alpha[:2] = [p*4/(6*np.pi)*mod, p/8*mod]
        alpha[2:] = -2*p*np.sin(np.pi*m/2)/(np.pi*m*(m**2 - 4))*mod

        # (in Ref. 2 Eq. 37.2, alpha is associated with the y-direction. Can someone explain?)
        return alpha

    def beta(self) -> NDArray[FOURIER_TERMS, complex]:
        r"""Fourier series coefficients modified for use in stress function equations

        Notes
        -----
        Exact solution:

        .. math:: \frac{-P}{2\pi}\int_{-\pi/2}^{\pi/2}\cos\theta\sin\theta\left(\cos m*\theta - i \sin m*\theta\right)\,d\theta
        .. math:: = -\frac{i P sin(\pi m/2)}{\pi (m^2-4)}

        Modifications to the Fourier series coefficients are developed in Eq. 37.2 [2]_

        Returns
        -------
        complex ndarray

        """
        h = self.h
        p = self.p
        N = self.FOURIER_TERMS
        m = np.arange(1, N + 1)

        # modification from Eq. 37.2 [2]_
        mod = 4 / (np.pi*m**2*h)

        beta = np.zeros(N, dtype=complex)
        beta[:2] = [-p*1j/(3*np.pi)*mod[0], -1j*p/8*mod[1]]
        beta[2:] = 1j*p*np.sin(np.pi*m[2:]/2)/(np.pi*(m[2:]**2 - 4))*mod[2:]

        # (in Ref. 2 Eq. 37.2, beta is associated with the x-direction. Can someone explain?)
        return beta

    def equilibrium_constants(self) -> tuple[float, float, float, float]:
        """Solve for constants of equilibrium

        When the plate has loads applied that are not in equilibrium, the unbalanced loads are reacted at infinity.
        This function solves for the constant terms in the stress functions that account for these reactions.

        Notes
        -----
        This method implements Eq. 37.5 [2]_. Complex terms have been expanded and resolved for
        A, A_bar, B and B_bar (setting Py equal to zero).

        Returns
        -------
        [A, A_bar, B, B_bar] : tuple
            real and imaginary parts of constants A and B
        """
        R1, R2 = np.real(self.mu1), np.imag(self.mu1)
        R3, R4 = np.real(self.mu2), np.imag(self.mu2)
        p = self.p
        h = self.h
        a11 = self.a[0, 0]
        a12 = self.a[0, 1]
        a22 = self.a[1, 1]
        a16 = self.a[0, 2]
        pi = np.pi

        mu_mat = np.array([[0., 1, 0., 1.],
                           [R2, R1, R4, R3],
                           [2*R1*R2, (R1**2 - R2**2), 2*R3*R4, (R3**2 - R4**2)],
                           [R2/(R1**2 + R2**2), -R1/(R1**2 + R2**2), R4/(R3**2 + R4**2), -R3/(R3**2 + R4**2)]])

        load_vec = p/(4.*pi*h) * np.array([0.,
                                           1.,
                                           a16/a11,
                                           a12/a22])

        A1, A2, B1, B2 = np.dot(np.linalg.inv(mu_mat), load_vec)
        return A1, A2, B1, B2

    def phi_1(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates the first stress function

        Notes
        -----
        This method implements [Eq. 37.3, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_2\alpha_m}{\mu_1-\mu_2}
        .. math:: \Phi_1=A\ln{\xi_1}+\sum_{m=1}^{\infty}\frac{C_m}{\xi_1^m}

        Parameters
        ----------
        z1 : ndarray
            1D complex array first mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        A = self.A + 1j * self.A_bar
        N = self.FOURIER_TERMS
        xi_1, sign_1 = self.xi_1(z1)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_1
        return np.array([(A*np.log(xi_1[i]) + np.sum((beta - mu2 * alpha) / (mu1 - mu2) / xi_1[i] ** m))
                         for i in range(len(xi_1))])

    def phi_2(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates the second stress function

        Notes
        -----
        This method implements [Eq. 37.3, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_1\alpha_m}{\mu_1-\mu_2}
        .. math:: \Phi_2=B\ln{\xi_2}-\sum_{m=1}^{\infty}\frac{m C_m}{\xi_2^m}

        Parameters
        ----------
        z2 : ndarray
            1D complex array second mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        B = self.B + 1j * self.B_bar
        N = self.FOURIER_TERMS
        xi_2, sign_2 = self.xi_2(z2)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_2
        return np.array([(B*np.log(xi_2[i]) - np.sum((beta - mu1 * alpha) / (mu1 - mu2) / xi_2[i] ** m))
                         for i in range(len(xi_2))])

    def phi_1_prime(self, z1: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates derivative of the first stress function

        Notes
        -----
        This method implements [Eq. 37.6, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_2\alpha_m}{\mu_1-\mu_2}
        .. math:: \eta_1=\pm\sqrt{z_1^2-a^2-\mu_1^2b^2}
        .. math:: \Phi_1'=-\frac{1}{\eta_1}(A-\sum_{m=1}^{\infty}\frac{m C_m}{\xi_1^m})

        Parameters
        ----------
        z1 : ndarray
            1D complex array first mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        a = self.r
        b = self.r
        A = self.A + 1j * self.A_bar
        N = self.FOURIER_TERMS
        xi_1, sign_1 = self.xi_1(z1)

        eta_1 = sign_1 * np.sqrt(z1 * z1 - a * a - b * b * mu1 * mu1)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_1
        return np.array([1 / eta_1[i] * (A - np.sum(m * (beta - mu2 * alpha) / (mu1 - mu2) / xi_1[i] ** m))
                        for i in range(len(xi_1))])

    def phi_2_prime(self, z2: NDArray[Any, complex]) -> NDArray[Any, complex]:
        r"""Calculates derivative of the second stress function

        Notes
        -----
        This method implements [Eq. 37.6, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_1\alpha_m}{\mu_1-\mu_2}
        .. math:: \eta_2=\pm\sqrt{z_2^2-a^2-\mu_2^2b^2}
        .. math:: \Phi_2'=-\frac{1}{\eta_2}(B+\sum_{m=1}^{\infty}\frac{m C_m}{\xi_2^m})

        Parameters
        ----------
        z2 : ndarray
            1D complex array second mapping parameter

        Returns
        -------
        ndarray
            1D complex array

        """
        mu1 = self.mu1
        mu2 = self.mu2
        a = self.r
        b = self.r
        B = self.B + 1j * self.B_bar
        N = self.FOURIER_TERMS
        xi_2, sign_2 = self.xi_2(z2)

        eta_2 = sign_2 * np.sqrt(z2 * z2 - a * a - b * b * mu2 * mu2)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_2
        return np.array([1 / eta_2[i] * (B + np.sum(m * (beta - mu1 * alpha) / (mu1 - mu2) / xi_2[i] ** m))
                         for i in range(len(xi_2))])
    
    def _cartesian_to_polar(self,  x: NDArray[Any, float], y: NDArray[Any, float])\
            -> tuple[NDArray[Any, float], NDArray[Any, float]]:
        """(Private method) Converts cartesian points to polar coordinates

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        radii : ndarray
            radius of each point
        angles : ndarray
            angle of each point

        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        r = np.sqrt(x**2 + y**2)

        # calculate angles and fix signs
        angles = np.arccos(np.array([1, 0]).dot(np.array([x, y])) / r)
        where_vals = np.nonzero(y)[0]
        angles[where_vals] = angles[where_vals] * np.sign(y[where_vals])

        return r, angles

        
    def _rotate_points(self, x: NDArray[Any, float], y: NDArray[Any, float])\
            -> tuple[NDArray[Any, float], NDArray[Any, float]]:
        """(Private method) Rotates points to account for bearing angle

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        x' : ndarray
            new x points
        y' : ndarray
            new y points

        """
        # rotation back to original coordinates
        rotation = -self.theta

        # convert points to polar coordinates
        r, angles = self._cartesian_to_polar(x, y)

        # rotate coordinates by negative theta
        angles += rotation

        # convert back to cartesian
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        return x, y

    def stress(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 3), float]:
        r""" Calculates the stress at (x, y) points in the plate

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        ndarray
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
            (n, 3) in-plane stress components in the cartesian coordinate system

        """
        # rotate points to account for bearing angle
        x, y = self._rotate_points(x, y)

        # calculate stresses and rotate back
        stresses = super().stress(x, y)
        return rotate_stress(stresses, angle=-self.theta)

    @_remove_bad_displacments
    def displacement(self, x: NDArray[Any, float], y: NDArray[Any, float]) -> NDArray[(Any, 2), float]:
        r""" Calculates the displacement at (x, y) points in the plate

        Notes
        -----
        This method implements Eq. 8.3 [2]_

        .. math:: u=2Re[p_1\Phi_1(z_1)+p_2\Phi_2(z_2)]
        .. math:: v=2Re[q_1\Phi_1(z_1)+q_2\Phi_2(z_2)]

        Parameters
        ----------
        x : array_like
            1D array x locations in the cartesian coordinate system
        y : array_like
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        ndarray
            [[u0, v0], [u1, v1], ... , [un, vn]]
            (n, 2) in-plane displacement components in the cartesian coordinate system

        """
        # rotate points to account for bearing angle
        x, y = self._rotate_points(x, y)
        return super().displacement(x, y)


