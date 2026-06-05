"""Utility functions supporting bolted joint analysis.

References
----------
.. [1] Whitney, J. M. and Nuismer, R. J. (1974) *Stress Fracture Criteria for
   Laminated Composites Containing Stress Concentrations*. Journal of Composite
   Materials, Vol. 8, pp. 253-265.
.. [2] Nuismer, R. J. and Whitney, J. M. (1975) *Uniaxial Failure of Composite
   Laminates Containing Stress Concentrations*. Fracture Mechanics of Composites,
   ASTM STP 593, pp. 117-142.
.. [3] Daniel, I. M. and Ishai, O. (2006) *Engineering Mechanics of Composite
   Materials* (2nd ed.). New York: Oxford University Press.

"""
import numpy as np
from nptyping import NDArray, Shape, Float

from bjsfm.lekhnitskii import rotate_material_matrix


def orthotropic_stress_concentration(a_inv: NDArray[Shape['3, 3'], Float], angle: float = 0.) -> float:
    r"""Stress concentration factor of an infinite orthotropic plate with a circular hole.

    Notes
    -----
    Implements the closed-form (Lekhnitskii) orthotropic stress concentration factor
    for an infinite plate with a circular hole, loaded in remote uniaxial tension
    (Eq. 4.51 [3]_):

    .. math::
        K_T^\infty = 1 + \sqrt{2\left(\sqrt{\tfrac{E_x}{E_y}} - \nu_{xy}\right)
        + \tfrac{E_x}{G_{xy}}}

    This is the hoop (tangential) stress concentration at the hole boundary
    **at** :math:`\theta = 90^\circ`, i.e. the point perpendicular to the applied
    load. This is the net-section location used by the Whitney-Nuismer point-stress
    criterion (see :func:`point_stress_ratio`), so it is the appropriate ``kt`` to
    pass to that function.

    The required engineering constants are taken from the laminate membrane
    compliance matrix (the inverse CLPT A-matrix, :math:`S = A^{-1}`) where
    :math:`E_x = 1/(t\,S_{11})`, :math:`E_y = 1/(t\,S_{22})`,
    :math:`G_{xy} = 1/(t\,S_{66})` and :math:`\nu_{xy} = -S_{12}/S_{11}`. The plate
    thickness cancels in every ratio, so only ``a_inv`` is required.

    For a quasi-isotropic laminate (:math:`E_x = E_y`,
    :math:`G_{xy} = E_x/[2(1 + \nu_{xy})]`) this reduces to the isotropic value
    :math:`K_T^\infty = 3`.

    Assumptions / limitations
    -------------------------
    * The laminate is **specially orthotropic** with respect to the load axis: the
      shear-extension coupling terms :math:`S_{16}, S_{26}` (``a_inv[0, 2]`` and
      ``a_inv[1, 2]``) are ignored. Use ``angle`` to align the load with a principal
      material axis where these terms vanish.
    * The result is the hoop stress at :math:`\theta = 90^\circ`. This equals the
      *maximum* hoop stress around the hole only when the peak occurs on-axis (true
      for principal-axis loading of typical balanced/symmetric laminates). For
      strongly anisotropic layups the true peak can shift off :math:`90^\circ` and
      slightly exceed this value.

    Parameters
    ----------
    a_inv : ndarray
        2D (3, 3) inverse CLPT A-matrix (membrane compliance), with the 0-degree
        material direction parallel to the x-axis
    angle : float, default 0.
        load direction measured counter-clockwise from the positive x-axis (radians);
        the compliance matrix is rotated so the load aligns with the x-axis

    Returns
    -------
    float
        orthotropic stress concentration factor :math:`K_T^\infty`

    """
    a_inv = np.asarray(a_inv, dtype=float)
    if angle:
        a_inv = rotate_material_matrix(a_inv, angle=angle)
    s11 = a_inv[0, 0]
    s22 = a_inv[1, 1]
    s12 = a_inv[0, 1]
    s66 = a_inv[2, 2]
    e_ratio = s22/s11        # E_x / E_y
    eg_ratio = s66/s11       # E_x / G_xy
    nu_xy = -s12/s11
    return 1 + np.sqrt(2*(np.sqrt(e_ratio) - nu_xy) + eg_ratio)


def point_stress_ratio(rc: float, radius: float, kt: float = 3.) -> float:
    r"""Notched-to-unnotched strength ratio from the point-stress criterion.

    Notes
    -----
    Implements the Whitney-Nuismer point-stress criterion for an infinite plate
    with a circular hole loaded in remote uniaxial tension (Eq. 11 [2]_). The normal
    stress ahead of the hole is

    .. math::
        \frac{\sigma_y(x, 0)}{\sigma^\infty} = \tfrac{1}{2}\left[2 + \xi^2
        + 3\xi^4 - (K_T^\infty - 3)\left(5\xi^6 - 7\xi^8\right)\right]

    where :math:`\xi = R / (R + r_c)` and :math:`R` is the hole radius. The
    point-stress criterion assumes failure occurs when the stress a distance
    :math:`r_c` ahead of the hole reaches the unnotched strength, giving the
    notched-to-unnotched strength ratio :math:`\sigma_N / \sigma_0` as the
    reciprocal of the expression above evaluated at :math:`x = R + r_c`.

    Parameters
    ----------
    rc : float
        characteristic distance (measured from the hole edge)
    radius : float
        hole radius
    kt : float, default 3.
        orthotropic (infinite-plate) stress concentration factor :math:`K_T^\infty`
        (use 3. for the isotropic case)

    Returns
    -------
    float
        notched-to-unnotched strength ratio :math:`\sigma_N / \sigma_0`

    """
    if rc < 0:
        raise ValueError("rc must be non-negative")
    if radius <= 0:
        raise ValueError("radius must be positive")
    xi = radius / (radius + rc)
    concentration = 1 + 0.5*xi**2 + 1.5*xi**4 - 0.5*(kt - 3)*(5*xi**6 - 7*xi**8)
    return 1./concentration


def characteristic_distance(notched_strength: float, unnotched_strength: float, radius: float,
                            kt: float = 3., tol: float = 1e-12, max_iter: int = 200) -> float:
    r"""Finds the point-stress characteristic distance from test strengths.

    Notes
    -----
    Back-calculates the Whitney-Nuismer characteristic distance :math:`r_c` (often
    written :math:`d_0`) by inverting the point-stress criterion (Eq. 11 [2]_), see
    :func:`point_stress_ratio`. The relationship is monotonic in :math:`r_c`, so a
    bisection root-find is used (avoiding a third-party solver dependency).

    The supplied ``notched_strength`` is the remote (gross-section) strength of an
    infinite plate with an open hole of the given ``radius``, and
    ``unnotched_strength`` is the strength of the same laminate without a hole.

    Parameters
    ----------
    notched_strength : float
        open-hole (notched) strength :math:`\sigma_N`
    unnotched_strength : float
        unnotched laminate strength :math:`\sigma_0`
    radius : float
        hole radius
    kt : float, default 3.
        orthotropic (infinite-plate) stress concentration factor :math:`K_T^\infty`
        (use 3. for the isotropic case)
    tol : float, default 1e-12
        absolute convergence tolerance on the strength ratio
    max_iter : int, default 200
        maximum number of bisection iterations

    Returns
    -------
    float
        characteristic distance :math:`r_c` (measured from the hole edge)

    Raises
    ------
    ValueError
        if the inputs are non-physical or the strength ratio is outside the range
        :math:`(1/K_T^\infty,\ 1)` reachable by the point-stress criterion

    """
    if unnotched_strength <= 0 or notched_strength <= 0:
        raise ValueError("strengths must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    target = notched_strength/unnotched_strength
    # at rc -> 0, ratio -> 1/kt (full stress concentration); at rc -> inf, ratio -> 1
    if not 1./kt < target < 1.:
        raise ValueError(
            f"notched/unnotched strength ratio {target:.4f} must lie in the open interval "
            f"(1/kt, 1) = ({1./kt:.4f}, 1.0) for a solution to exist"
        )

    def residual(rc: float) -> float:
        return point_stress_ratio(rc, radius, kt=kt) - target

    # bracket the root: ratio increases monotonically with rc, find an upper bound
    lo, hi = 0., radius
    while residual(hi) < 0:
        hi *= 2
        if hi > radius*1e12:
            raise ValueError("failed to bracket a characteristic distance")

    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        res = residual(mid)
        if abs(res) < tol:
            return mid
        if res < 0:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)
