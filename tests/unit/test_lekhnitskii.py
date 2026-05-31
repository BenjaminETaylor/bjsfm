"""Unit tests for the pure-Python machinery in :mod:`bjsfm.lekhnitskii`.

These tests do not depend on the compiled Fortran reference; they exercise the rotation helpers,
mapping invariants, single-/multi-point input handling, and edge-case branches directly.
"""
import logging
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from bjsfm.lekhnitskii import (
    rotate_stress,
    rotate_strain,
    rotate_material_matrix,
    rotate_complex_parameters,
    UnloadedHole,
    LoadedHole,
)
from tests.test_data import (
    QUASI_INV, SOFT_INV, HARD_INV, QUASI_THICK, DIAMETER,
)


class RotateStressTests(unittest.TestCase):

    def test_zero_angle_is_identity(self):
        stresses = np.array([[123., -45., 67.]])
        assert_array_almost_equal(rotate_stress(stresses, angle=0.), stresses)

    def test_uniaxial_to_90_degrees(self):
        # sigma_x only, rotated 90 deg -> becomes sigma_y
        assert_array_almost_equal(
            rotate_stress(np.array([[1., 0., 0.]]), angle=np.pi / 2),
            np.array([[0., 1., 0.]]),
        )

    def test_pure_shear_to_45_degrees(self):
        # pure shear rotated 45 deg -> biaxial [tau, -tau, 0]
        tau = 5.
        assert_array_almost_equal(
            rotate_stress(np.array([[0., 0., tau]]), angle=np.pi / 4),
            np.array([[tau, -tau, 0.]]),
        )

    def test_round_trip(self):
        stresses = np.array([[10., -3., 4.], [1., 2., 3.]])
        for angle in np.deg2rad([15., 45., 90., 137., -60.]):
            rotated = rotate_stress(stresses, angle=angle)
            assert_array_almost_equal(rotate_stress(rotated, angle=-angle), stresses)


class RotateStrainTests(unittest.TestCase):

    def test_zero_angle_is_identity(self):
        strains = np.array([[1.2e-3, -4.5e-4, 6.7e-4]])
        assert_array_almost_equal(rotate_strain(strains, angle=0.), strains)

    def test_uniaxial_to_90_degrees(self):
        assert_array_almost_equal(
            rotate_strain(np.array([[1., 0., 0.]]), angle=np.pi / 2),
            np.array([[0., 1., 0.]]),
        )

    def test_round_trip(self):
        strains = np.array([[1e-3, -2e-3, 3e-3], [4e-4, 5e-4, -6e-4]])
        for angle in np.deg2rad([15., 45., 90., 137., -60.]):
            rotated = rotate_strain(strains, angle=angle)
            assert_array_almost_equal(rotate_strain(rotated, angle=-angle), strains)


class RotateMaterialMatrixTests(unittest.TestCase):

    def test_zero_angle_is_identity(self):
        assert_array_almost_equal(rotate_material_matrix(QUASI_INV, angle=0.), QUASI_INV)

    def test_180_degrees_is_identity(self):
        # a full half-turn leaves the orthotropic compliance matrix unchanged
        assert_array_almost_equal(rotate_material_matrix(QUASI_INV, angle=np.pi), QUASI_INV)

    def test_result_is_symmetric(self):
        for a_inv in (QUASI_INV, SOFT_INV, HARD_INV):
            for angle in np.deg2rad([30., 45., 75., 110.]):
                rotated = rotate_material_matrix(a_inv, angle=angle)
                assert_array_almost_equal(rotated, rotated.T)

    def test_invariants_preserved(self):
        # Eq. 9.7 [2]_ invariants checked internally; verify here as well
        for a_inv in (QUASI_INV, SOFT_INV, HARD_INV):
            a11, a12, a22, a66 = a_inv[0, 0], a_inv[0, 1], a_inv[1, 1], a_inv[2, 2]
            for angle in np.deg2rad([10., 33., 90., 145.]):
                r = rotate_material_matrix(a_inv, angle=angle)
                assert_almost_equal(r[0, 0] + r[1, 1] + 2 * r[0, 1], a11 + a22 + 2 * a12, decimal=4)
                assert_almost_equal(r[2, 2] - 4 * r[0, 1], a66 - 4 * a12, decimal=4)


class RotateComplexParametersTests(unittest.TestCase):

    def test_zero_angle_is_identity(self):
        hole = UnloadedHole([0., 0., 0.], DIAMETER, QUASI_THICK, QUASI_INV)
        mu1p, mu2p = rotate_complex_parameters(hole.mu1, hole.mu2, angle=0.)
        self.assertAlmostEqual(mu1p, hole.mu1)
        self.assertAlmostEqual(mu2p, hole.mu2)

    def test_consistent_with_rotated_material_roots(self):
        # roots of the rotated compliance matrix should match the rotated complex parameters
        for a_inv in (QUASI_INV, SOFT_INV, HARD_INV):
            base = UnloadedHole([0., 0., 0.], DIAMETER, QUASI_THICK, a_inv)
            for angle in np.deg2rad([20., 45., 90., 115.]):
                rotated_hole = UnloadedHole([0., 0., 0.], DIAMETER, QUASI_THICK,
                                            rotate_material_matrix(a_inv, angle=angle))
                mu1p, mu2p = rotate_complex_parameters(base.mu1, base.mu2, angle=angle)
                # roots are returned as an unordered conjugate set; impose a canonical ordering
                # (imag first, then real as a tiebreaker for symmetric roots) before comparing
                key = lambda c: (round(c.imag, 6), round(c.real, 6))
                expected = sorted([rotated_hole.mu1, rotated_hole.mu2], key=key)
                actual = sorted([mu1p, mu2p], key=key)
                assert_array_almost_equal(actual, expected, decimal=5)


class RootsTests(unittest.TestCase):

    def test_roots_are_conjugate_pairs_with_positive_imag(self):
        for a_inv in (QUASI_INV, SOFT_INV, HARD_INV):
            hole = UnloadedHole([0., 0., 0.], DIAMETER, QUASI_THICK, a_inv)
            self.assertGreater(np.imag(hole.mu1), 0.)
            self.assertGreater(np.imag(hole.mu2), 0.)
            self.assertAlmostEqual(hole.mu1, np.conj(hole.mu1_bar))
            self.assertAlmostEqual(hole.mu2, np.conj(hole.mu2_bar))


class MappingTests(unittest.TestCase):

    def test_xi_magnitude_is_one_on_hole_boundary(self):
        # On the hole boundary the conformal map should satisfy |xi| == 1
        hole = UnloadedHole([100., 50., 25.], DIAMETER, QUASI_THICK, QUASI_INV)
        thetas = np.linspace(0, 2 * np.pi, num=37, endpoint=False)
        x = hole.r * np.cos(thetas)
        y = hole.r * np.sin(thetas)
        z1 = x + hole.mu1 * y
        z2 = x + hole.mu2 * y
        xi_1s, sign_1s = hole.xi_1(z1)
        xi_2s, sign_2s = hole.xi_2(z2)
        assert_array_almost_equal(np.abs(xi_1s), np.ones_like(thetas), decimal=5)
        assert_array_almost_equal(np.abs(xi_2s), np.ones_like(thetas), decimal=5)
        # every boundary point must have a valid sign (no unmapped zeros)
        self.assertTrue(np.all(np.abs(sign_1s) == 1))
        self.assertTrue(np.all(np.abs(sign_2s) == 1))

    def test_unsolvable_point_logs_warning(self):
        # A point at the hole centre cannot be mapped (|xi| < 1 for both signs) -> warning
        hole = UnloadedHole([100., 0., 0.], DIAMETER, QUASI_THICK, QUASI_INV)
        z1 = np.array([0. + 0j])
        with self.assertLogs('bjsfm.lekhnitskii', level=logging.WARNING):
            hole.xi_1(z1)


class SinglePointInputTests(unittest.TestCase):

    def test_unloaded_single_point_stress_shape(self):
        hole = UnloadedHole([100., 0., 0.], DIAMETER, QUASI_THICK, QUASI_INV)
        stress = hole.stress([hole.r], [0.])
        self.assertEqual(stress.shape, (1, 3))

    def test_unloaded_single_point_displacement_shape(self):
        hole = UnloadedHole([100., 0., 0.], DIAMETER, QUASI_THICK, QUASI_INV)
        disp = hole.displacement([hole.r], [0.])
        self.assertEqual(disp.shape, (1, 2))

    def test_loaded_single_point_stress_shape(self):
        hole = LoadedHole(100., DIAMETER, QUASI_THICK, QUASI_INV, theta=0.)
        stress = hole.stress([hole.r], [0.])
        self.assertEqual(stress.shape, (1, 3))

    def test_multi_point_matches_single_point(self):
        # stress at a point should be identical whether computed alone or within a batch
        hole = UnloadedHole([100., -40., 30.], DIAMETER, QUASI_THICK, QUASI_INV)
        thetas = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        x = (hole.r + 0.1) * np.cos(thetas)
        y = (hole.r + 0.1) * np.sin(thetas)
        batch = hole.stress(x, y)
        for i in range(len(thetas)):
            single = hole.stress([x[i]], [y[i]])
            np.testing.assert_allclose(single[0], batch[i], rtol=1e-6, atol=1e-3)


class BadDisplacementTests(unittest.TestCase):

    def test_point_behind_bearing_is_nan(self):
        # for theta=0 the unloadable point sits at angle pi (i.e. (-r, 0))
        hole = LoadedHole(100., DIAMETER, QUASI_THICK, QUASI_INV, theta=0.)
        disp = hole.displacement([-hole.r], [0.])
        self.assertTrue(np.all(np.isnan(disp[0])))

    def test_point_in_front_of_bearing_is_finite(self):
        hole = LoadedHole(100., DIAMETER, QUASI_THICK, QUASI_INV, theta=0.)
        disp = hole.displacement([hole.r], [0.])
        self.assertTrue(np.all(np.isfinite(disp[0])))


if __name__ == '__main__':
    unittest.main()




