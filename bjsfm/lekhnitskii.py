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
import numpy as np
import numpy.testing as npt
import bjsfm.fourier_series as fs


logger = logging.getLogger(__name__)


def rotate_plane_stress(stresses, angle=0.):
    r"""Rotates the stress components by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    stresses : ndarray
        2D array of [:math: `\sigma_x, \sigma_y, \tau_{xy}`] in-plane stresses
    angle : float optional
        angle measured counter-clockwise from positive x-axis (radians), defaults=0

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
    return np.array([rotation_matrix.dot(stresses[i]) for i in range(len(stresses))])


def rotate_material_matrix(a_inv, angle=0.):
    r"""Rotates the material compliance matrix by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Notes
    -----
    This function implements Eq. 9.6 [1]_

    Parameters
    ----------
    a_inv : ndarray
        2D (3, 3) inverse CLPT A-matrix
    angle : float
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
    npt.assert_almost_equal(a11p + a22p + 2*a12p, a11 + a22 + 2*a12, decimal=4)
    npt.assert_almost_equal(a66p - 4*a12p, a66 - 4*a12, decimal=4)

    return np.array([[a11p, a12p, a16p], [a12p, a22p, a26p], [a16p, a26p, a66p]])


def rotate_complex_parameters(mu1, mu2, angle=0.):
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
    angle : float optional
        angle measured counter-clockwise from positive x-axis (radians), defaults=0

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

    def __init__(self, diameter, thickness, a_inv):
        self.r = diameter/2.
        self.a = np.array(a_inv)
        self.h = thickness
        self.mu1, self.mu2, self.mu1_bar, self.mu2_bar = self.roots()

    def roots(self):
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

    def xi_1(self, z1s):
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

    def xi_2(self, z2s):
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
    def phi_1_prime(self, z1):
        raise NotImplementedError("You must implement this function.")

    @abc.abstractmethod
    def phi_2_prime(self, z2):
        raise NotImplementedError("You must implement this function.")

    def stress(self, x, y):
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

        x = np.array(x)
        y = np.array(y)

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1_prime = self.phi_1_prime(z1)
        phi_2_prime = self.phi_2_prime(z2)

        sx = 2.0 * np.real(mu1 * mu1 * phi_1_prime + mu2 * mu2 * phi_2_prime)
        sy = 2.0 * np.real(phi_1_prime + phi_2_prime)
        sxy = -2.0 * np.real(mu1 * phi_1_prime + mu2 * phi_2_prime)

        return np.array([sx, sy, sxy]).T


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

    def __init__(self, loads, diameter, thickness, a_inv):
        super().__init__(diameter, thickness, a_inv)
        self.applied_stress = np.array(loads) / self.h

    def alpha(self):
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

    def beta(self):
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

    def phi_1_prime(self, z1):
        r"""Calculates derivative of the first stress function

        Notes
        -----
        This method implements Eq. A.8 [1]_

        .. math:: C_1=\frac{\beta_1-\mu_2\alpha_1}{\mu_1-\mu_2}
        .. math:: \eta_1=\pm\sqrt{z_1^2-a^2-\mu_1^2b^2}
        .. math:: \kappa_1=\frac{1}{a-i\mu_1b}
        .. math:: \Phi_1'=-\frac{C_1}{\xi_1^2}(1+\frac{z_1}{\eta_1}\kappa_1

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

    def phi_2_prime(self, z2):
        r"""Calculates derivative of the second stress function

        Notes
        -----
        This method implements Eq. A.8 [1]_

        .. math:: C_2=-\frac{\beta_1-\mu_1\alpha_1}{\mu_1-\mu_2}
        .. math:: \eta_2=\pm\sqrt{z_2^2-a^2-\mu_2^2b^2}
        .. math:: \kappa_2=\frac{1}{a-i\mu_2b}
        .. math:: \Phi_2'=-\frac{C_2}{\xi_2^2}(1+\frac{z_2}{\eta_2}\kappa_2

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

    def stress(self, x, y):
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


class LoadedHole(Hole):
    """Class for defining a loaded hole in an infinite anisotropic homogeneous plate

    A cosine bearing load distribution is assumed to apply to the inside of the hole.

    Notes
    -----
    Bearing distribution as shown below Ref. [4]_

    .. image:: /docs/img/cosine_distribution.png

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

    # X_DIR_COEFFICIENTS = self._x_dir_fourier_coefficients()
    X_DIR_COEFFICIENTS = np.array([
        2.12206591e-01 - 4.77083644e-17j, 1.25000000e-01 - 5.89573465e-17j,
        4.24413182e-02 - 1.91840853e-17j, -8.90314393e-18 - 1.79348322e-19j,
        -6.06304545e-03 + 6.55633890e-18j, 5.48463980e-18 + 4.37501201e-18j,
        2.02101515e-03 - 3.66997376e-18j, -2.47147905e-18 - 3.77237815e-19j,
        -9.18643250e-04 + 6.67550845e-19j, 1.15294597e-18 + 4.32409913e-20j,
        4.94654058e-04 - 5.26048781e-18j, -1.92490138e-18 - 3.55274303e-18j,
        -2.96792435e-04 + 4.00276461e-18j, 3.49945789e-18 + 2.84432075e-18j,
        1.92042164e-04 - 7.15349518e-19j, -2.10847715e-18 + 5.86429928e-19j,
        -1.31397270e-04 + 5.42357122e-19j, 5.26279974e-19 + 5.07907945e-19j,
        9.38551927e-05 - 1.60287068e-18j, -2.62667554e-19 - 2.81642867e-20j,
        -6.93712294e-05 + 4.72318710e-19j, -1.55309233e-19 - 6.73163746e-19j,
        5.27221344e-05 + 3.74419334e-19j, 1.10507308e-18 - 3.45051024e-18j,
        -4.10061045e-05 + 1.56923065e-19j, 9.40356979e-19 - 2.19017030e-18j,
        3.25220829e-05 - 3.91078386e-19j, 1.36872347e-19 - 4.27353360e-19j,
        -2.62274862e-05 + 2.86611820e-19j, 9.78311008e-20 - 7.89061684e-20j,
        2.14588523e-05 - 8.91027872e-19j, -1.30904740e-19 + 1.91919825e-19j,
        -1.77801919e-05 + 1.97944104e-19j, 8.14254172e-19 + 2.81801032e-19j,
        1.48969176e-05 - 1.66624951e-19j, -1.34123974e-18 + 1.17525380e-18j,
        -1.26050841e-05 + 1.21462369e-18j, 5.21951371e-19 - 1.06955735e-18j,
        1.07604376e-05 - 1.17456794e-18j, -8.16624019e-20 + 5.13214752e-20j,
        -9.25898123e-06 - 1.65297614e-19j, 3.30062278e-19 - 2.46250926e-20j,
        8.02445040e-06 - 2.73275116e-19j, -2.39245061e-19 + 5.01995076e-19j,
        -7.00005248e-06 + 1.01720924e-19j
    ])

    # Y_DIR_COEFFICIENTS = self._y_dir_fourier_coefficients()
    Y_DIR_COEFFICIENTS = np.array([
        -1.94319243e-17 - 1.06103295e-01j, -5.45839291e-17 - 1.25000000e-01j,
        -3.62876318e-17 - 6.36619772e-02j, 1.30591839e-18 + 1.52792630e-17j,
        1.58336660e-17 + 1.51576136e-02j, 1.61007420e-18 - 1.20107231e-17j,
        -9.15844587e-18 - 7.07355303e-03j, -4.65834606e-19 + 4.69348027e-18j,
        7.82631893e-18 + 4.13389463e-03j, -2.07168349e-19 - 5.48019331e-18j,
        -7.79806861e-18 - 2.72059732e-03j, -8.28820898e-19 + 3.72983658e-18j,
        5.67464898e-18 + 1.92915083e-03j, -9.41779078e-19 - 2.96224847e-18j,
        -4.81136247e-18 - 1.44031623e-03j, -4.18882423e-20 + 3.92096760e-18j,
        3.53379639e-18 + 1.11687679e-03j, 1.18208219e-18 - 3.45316542e-18j,
        -3.35800239e-18 - 8.91624331e-04j, -3.88844853e-19 + 2.81568924e-18j,
        3.55287198e-18 + 7.28397909e-04j, -7.24302864e-22 - 3.24725934e-18j,
        -2.86484044e-18 - 6.06304545e-04j, 1.85812997e-18 + 2.72227446e-18j,
        2.71489222e-18 + 5.12576306e-04j, -1.22325211e-18 - 2.62305288e-18j,
        -3.25375118e-18 - 4.39048119e-04j, 5.06148684e-20 + 1.30612327e-18j,
        2.02547194e-18 + 3.80298550e-04j, -1.10424267e-19 - 1.61508137e-18j,
        -2.30407373e-18 - 3.32612211e-04j, -4.65115570e-19 + 1.28879601e-18j,
        2.22873521e-18 + 2.93373167e-04j, 8.28830477e-20 - 1.39232809e-18j,
        -1.82653809e-18 - 2.60696058e-04j, 3.63246046e-19 + 1.92275788e-18j,
        1.97581297e-18 + 2.33194056e-04j, 2.19814138e-20 - 1.77673402e-18j,
        -1.35481930e-18 - 2.09828534e-04j, 9.33755027e-20 + 1.34376519e-18j,
        1.71339592e-18 + 1.89809115e-04j, 1.30928047e-19 - 1.79294538e-18j,
        -1.94173495e-18 - 1.72525684e-04j, -1.07013407e-19 + 9.92738558e-19j,
        1.57354012e-18 + 1.57501181e-04j
    ])

    def __init__(self, load, diameter, thickness, a_inv, theta=0.):
        a_inv = rotate_material_matrix(a_inv, angle=theta)
        super().__init__(diameter, thickness, a_inv)
        self.p = load
        self.theta = theta
        self.A, self.A_bar, self.B, self.B_bar = self.equilibrium_constants()

    def _x_dir_fourier_coefficients(self, sample_rate=100000):
        """Calculates Fourier coefficients of x-direction components of bearing load

        This function calculates the fourier series coefficients for the x-direction components of a cosine bearing
        load distribution centered on the positive x-axis.

        Parameters
        ----------
        sample_rate : int, optional
            used to tune the fast fourier transform (FFT) algorithm for accuracy

        Returns
        -------
        complex ndarray
            fourier series coefficients
        """
        N = self.FOURIER_TERMS

        def brg_load_x_component(thetas):
            """x-direction components of a cosine load distribution centered at positive x-axis

            Parameters
            ----------
            thetas : 1D ndarray
                angles

            Returns
            -------
            ndarray
                array of x-direction force terms for each angle in thetas
            """
            new_array = np.zeros(len(thetas))
            for i, angle in enumerate(thetas):
                if -np.pi / 2 <= angle <= np.pi / 2:
                    # x-direction component of cosine load distribution
                    new_array[i] = np.cos(angle) ** 2
            return new_array

        # return all coefficients except the first one (Ao)
        return fs.fourier_series_coefficients(brg_load_x_component, 2 * np.pi, N, sample_rate=sample_rate)[1:]

    def _y_dir_fourier_coefficients(self, sample_rate=100000):
        """Calculates Fourier coefficients of y-direction components of bearing load

        This function calculates the fourier series coefficients for the y-direction components of a cosine bearing
        load distribution centered on the positive x-axis.

        Parameters
        ----------
        sample_rate : int, optional
            used to tune the fast fourier transform (FFT) algorithm for accuracy

        Returns
        -------
        complex ndarray
            fourier series coefficients
        """
        N = self.FOURIER_TERMS

        def brg_load_y_component(thetas):
            """Y-direction components of a cosine load distribution centered at positive x-axis

            Parameters
            ----------
            thetas : ndarray
                angles (radians)

            Returns
            -------
            ndarray
                array of y-direction force terms for each angle in thetas
            """
            new_array = np.zeros(len(thetas))
            for i, angle in enumerate(thetas):
                if -np.pi / 2 <= angle <= np.pi / 2:
                    # y-direction component of cosine load distribution
                    new_array[i] = np.cos(angle) * np.sin(angle)
            return new_array

        # return all coefficients except the first one (Ao)
        return fs.fourier_series_coefficients(brg_load_y_component, 2 * np.pi, N, sample_rate=sample_rate)[1:]

    def alphas(self):
        """Fourier series coefficients modified for use in stress function equations

        Notes
        -----
        Modifications to the Fourier series coefficients are developed in Eq. 37.2 [2]_

        Returns
        -------
        complex ndarray

        """
        h = self.h
        p = self.p

        # (in Ref. 2 Eq. 37.2, alpha is associated with the y-direction. Can someone explain?)
        # return -p / (np.pi * h) * self._x_dir_fourier_coefficients()
        # hard coded alpha values used to speed up runtime
        return -p / (np.pi * h) * self.X_DIR_COEFFICIENTS

    def betas(self):
        """Fourier series coefficients modified for use in stress function equations

        Notes
        -----
        Modifications to the Fourier series coefficients are developed in Eq. 37.2 [2]_

        Returns
        -------
        complex ndarray

        """
        h = self.h
        p = self.p
        N = self.FOURIER_TERMS
        m = np.arange(1, N + 1)

        # (in Ref. 2 Eq. 37.2, beta is associated with the x-direction. Can someone explain?)
        # return 4 * p / (np.pi * m**2 * h) * self._y_dir_fourier_coefficients()
        # hard coded beta values used to speed up runtime
        return 4 * p / (np.pi * m**2 * h) * self.Y_DIR_COEFFICIENTS

    def equilibrium_constants(self):
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

    def phi_1_prime(self, z1):
        r"""Calculates derivative of the first stress function

        Notes
        -----
        This method implements [Eq. 37.6, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_2\alpha_m}{\mu_1-\mu_2}
        .. math:: \eta_1=\pm\sqrt{z_1^2-a^2-\mu_1^2b^2}
        .. math:: \Phi_1'=-\frac{1}{\eta_1}(A-\sum_{m=1}^{\infty}\frac{C_m}{\xi_1^m})

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
        alphas = self.alphas()
        betas = self.betas()

        return np.array([1 / eta_1[i] * (A - np.sum(m * (betas - mu2 * alphas) / (mu1 - mu2) / xi_1[i] ** m))
                        for i in range(len(xi_1))])

    def phi_2_prime(self, z2):
        r"""Calculates derivative of the first stress function

        Notes
        -----
        This method implements [Eq. 37.6, Ref. 2]

        .. math:: C_m=\frac{\beta_m-\mu_1\alpha_m}{\mu_1-\mu_2}
        .. math:: \eta_2=\pm\sqrt{z_2^2-a^2-\mu_2^2b^2}
        .. math:: \Phi_2'=-\frac{1}{\eta_2}(B+\sum_{m=1}^{\infty}\frac{C_m}{\xi_2^m})

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
        alphas = self.alphas()
        betas = self.betas()

        return np.array([1 / eta_2[i] * (B + np.sum(m * (betas - mu1 * alphas) / (mu1 - mu2) / xi_2[i] ** m))
                         for i in range(len(xi_2))])

    def stress(self, x, y):
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
        # rotation back to original coordinates
        rotation = -self.theta

        # convert points to polar coordinates
        x = np.array(x)
        y = np.array(y)
        r = np.sqrt(x**2 + y**2)
        # calculate angles and fix signs
        angles = np.arccos(np.array([1, 0]).dot(np.array([x, y])) / r)
        where_vals = np.nonzero(y)[0]
        angles[where_vals] = angles[where_vals] * np.sign(y[where_vals])

        # rotate coordinates by negative theta
        angles += rotation

        # convert back to cartesian
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        # calculate stresses and rotate
        stresses = super().stress(x, y)
        return rotate_plane_stress(stresses, angle=rotation)



