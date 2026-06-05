"""Unit tests for :mod:`bjsfm.utilities`.

These tests exercise the Whitney-Nuismer point-stress helpers: the forward strength-ratio
model and the inverse characteristic-distance solver, including their analytic limits and
input-validation branches.
"""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from bjsfm.lekhnitskii import UnloadedHole
from bjsfm.utilities import point_stress_ratio, characteristic_distance, orthotropic_stress_concentration
from tests.test_data import (
    QUASI_INV, HARD_INV, SOFT_INV, QUASI_THICK, HARD_THICK, DIAMETER,
)


def _isotropic_compliance(e: float = 10e6, nu: float = 0.3) -> np.ndarray:
    """Membrane compliance (inverse A-matrix form) for an isotropic material."""
    g = e/(2*(1 + nu))
    return np.array([
        [1/e, -nu/e, 0.],
        [-nu/e, 1/e, 0.],
        [0., 0., 1/g],
    ])


def _hoop_kt_at_90(a_inv: np.ndarray, t: float, angle: float = 0.) -> float:
    """Hoop-stress concentration at theta=90deg from the rigorous Lekhnitskii solution.

    Builds an infinite-plate solution with a unit far-field stress aligned with ``angle``
    and returns the tangential stress at the boundary point perpendicular to the load.
    """
    d = DIAMETER
    # unit far-field stress (sigma = N / t) rotated to the requested load direction
    sx, sy, sxy = np.array([1.0, 0., 0.])
    c, s = np.cos(angle), np.sin(angle)
    nx = (sx*c**2 + sy*s**2 - 2*sxy*s*c)*t
    ny = (sx*s**2 + sy*c**2 + 2*sxy*s*c)*t
    nxy = ((sx - sy)*s*c + sxy*(c**2 - s**2))*t
    hole = UnloadedHole([nx, ny, nxy], d, t, a_inv)
    # boundary point 90deg from the load direction
    th = angle + np.pi/2
    r = d/2
    stress = hole.stress([r*np.cos(th)], [r*np.sin(th)])[0]
    # hoop stress at that boundary point (tangent direction == load direction)
    return stress[0]*np.cos(angle)**2 + stress[1]*np.sin(angle)**2 + 2*stress[2]*np.sin(angle)*np.cos(angle)


class OrthotropicStressConcentrationTests(unittest.TestCase):

    def test_matches_lekhnitskii_solution(self):
        # the closed-form must equal the rigorous Lekhnitskii hoop stress at theta=90deg
        for a_inv, t in [(QUASI_INV, QUASI_THICK), (HARD_INV, HARD_THICK)]:
            for angle in (0., np.pi/2):
                assert_almost_equal(
                    orthotropic_stress_concentration(a_inv, angle=angle),
                    _hoop_kt_at_90(a_inv, t, angle=angle),
                    decimal=4,
                )

    def test_isotropic_is_three(self):
        # isotropic SCF must equal 3 regardless of Poisson's ratio
        for nu in (0.0, 0.25, 0.3, 0.45):
            self.assertAlmostEqual(orthotropic_stress_concentration(_isotropic_compliance(nu=nu)), 3.)

    def test_quasi_isotropic_is_three(self):
        # a quasi-isotropic laminate has the isotropic value
        assert_almost_equal(orthotropic_stress_concentration(QUASI_INV), 3., decimal=6)

    def test_quasi_isotropic_invariant_under_rotation(self):
        # quasi-isotropic in-plane response is rotationally invariant
        for angle in np.deg2rad([15., 30., 45., 90.]):
            assert_almost_equal(orthotropic_stress_concentration(QUASI_INV, angle=angle), 3., decimal=6)

    def test_stiff_direction_has_higher_kt(self):
        # HARD laminate is stiffer along x -> higher SCF than transverse
        kt_x = orthotropic_stress_concentration(HARD_INV)
        kt_y = orthotropic_stress_concentration(HARD_INV, angle=np.pi/2)
        self.assertGreater(kt_x, kt_y)
        self.assertGreater(kt_x, 3.)
        self.assertLess(kt_y, 3.)

    def test_90_degree_rotation_swaps_axes(self):
        # rotating the load 90 deg is equivalent to swapping Ex and Ey
        kt_y = orthotropic_stress_concentration(HARD_INV, angle=np.pi/2)
        swapped = HARD_INV.copy()
        swapped[0, 0], swapped[1, 1] = HARD_INV[1, 1], HARD_INV[0, 0]
        assert_almost_equal(orthotropic_stress_concentration(swapped), kt_y, decimal=6)

    def test_accepts_array_like_input(self):
        # plain nested lists should work (not only ndarrays)
        kt_arr = orthotropic_stress_concentration(SOFT_INV)
        kt_list = orthotropic_stress_concentration(SOFT_INV.tolist())
        assert_almost_equal(kt_arr, kt_list, decimal=10)

    def test_feeds_point_stress_ratio(self):
        # SCF integrates with the point-stress criterion: ratio -> 1/kt at the hole edge
        kt = orthotropic_stress_concentration(HARD_INV)
        assert_almost_equal(point_stress_ratio(0., 0.125, kt=kt), 1./kt, decimal=10)


class PointStressRatioTests(unittest.TestCase):

    def test_hole_edge_limit_is_one_over_kt(self):
        # rc -> 0 : full stress concentration, ratio -> 1/kt
        for kt in (3., 4., 5.):
            self.assertAlmostEqual(point_stress_ratio(0., 0.125, kt=kt), 1./kt)

    def test_far_field_limit_approaches_one(self):
        # rc -> inf : no concentration felt, ratio -> 1
        self.assertAlmostEqual(point_stress_ratio(1e9, 0.125), 1., places=6)

    def test_monotonically_increasing_with_distance(self):
        radius = 0.125
        rcs = np.linspace(0., 1., 50)
        ratios = [point_stress_ratio(rc, radius) for rc in rcs]
        self.assertTrue(np.all(np.diff(ratios) > 0))

    def test_isotropic_kt_default_is_three(self):
        radius = 0.125
        rc = 0.02
        self.assertEqual(point_stress_ratio(rc, radius), point_stress_ratio(rc, radius, kt=3.))

    def test_ratio_within_physical_bounds(self):
        radius = 0.125
        for rc in (0.001, 0.01, 0.05, 0.2, 1.0):
            ratio = point_stress_ratio(rc, radius, kt=4.)
            self.assertGreater(ratio, 1./4.)
            self.assertLess(ratio, 1.)

    def test_negative_rc_raises(self):
        with self.assertRaises(ValueError):
            point_stress_ratio(-0.01, 0.125)

    def test_non_positive_radius_raises(self):
        with self.assertRaises(ValueError):
            point_stress_ratio(0.01, 0.)
        with self.assertRaises(ValueError):
            point_stress_ratio(0.01, -0.125)


class CharacteristicDistanceTests(unittest.TestCase):

    def test_round_trip_isotropic(self):
        # forward model then inverse should recover the original distance
        radius = 0.125
        rc_true = 0.015
        ratio = point_stress_ratio(rc_true, radius)
        rc = characteristic_distance(ratio, 1.0, radius)
        assert_almost_equal(rc, rc_true, decimal=8)

    def test_round_trip_orthotropic(self):
        radius = 0.125
        rc_true = 0.022
        for kt in (4., 5., 6.):
            ratio = point_stress_ratio(rc_true, radius, kt=kt)
            rc = characteristic_distance(ratio, 1.0, radius, kt=kt)
            assert_almost_equal(rc, rc_true, decimal=8)

    def test_uses_strength_ratio_not_absolute_values(self):
        # only the ratio matters; scaling both strengths leaves rc unchanged
        radius = 0.125
        rc_a = characteristic_distance(40e3, 100e3, radius)
        rc_b = characteristic_distance(4., 10., radius)
        assert_almost_equal(rc_a, rc_b, decimal=10)

    def test_solution_satisfies_criterion(self):
        radius = 0.1
        notched, unnotched = 45e3, 100e3
        rc = characteristic_distance(notched, unnotched, radius)
        assert_almost_equal(point_stress_ratio(rc, radius), notched/unnotched, decimal=10)

    def test_larger_radius_gives_larger_distance(self):
        # for a fixed strength ratio, rc scales with hole radius
        ratio = 0.5
        rc_small = characteristic_distance(ratio, 1.0, 0.1)
        rc_large = characteristic_distance(ratio, 1.0, 0.2)
        self.assertGreater(rc_large, rc_small)

    def test_ratio_below_one_over_kt_raises(self):
        # ratio <= 1/kt is unreachable (would require rc < 0)
        with self.assertRaises(ValueError):
            characteristic_distance(0.2, 1.0, 0.125)  # 0.2 < 1/3

    def test_ratio_at_or_above_one_raises(self):
        with self.assertRaises(ValueError):
            characteristic_distance(1.0, 1.0, 0.125)
        with self.assertRaises(ValueError):
            characteristic_distance(1.5, 1.0, 0.125)

    def test_non_positive_strengths_raise(self):
        with self.assertRaises(ValueError):
            characteristic_distance(0., 1.0, 0.125)
        with self.assertRaises(ValueError):
            characteristic_distance(0.5, 0., 0.125)
        with self.assertRaises(ValueError):
            characteristic_distance(-0.5, 1.0, 0.125)

    def test_non_positive_radius_raises(self):
        with self.assertRaises(ValueError):
            characteristic_distance(0.5, 1.0, 0.)
        with self.assertRaises(ValueError):
            characteristic_distance(0.5, 1.0, -0.125)


if __name__ == '__main__':
    unittest.main()

