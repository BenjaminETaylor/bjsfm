import unittest
import numpy as np


class LoadedHoleTests(unittest.TestCase):

    # points to test at must be equally spaced around hole, starting at zero degrees
    TEST_POINTS = ((0.125, 0.), (0., 0.125), (-0.125, 0.), (0., -0.125))
    SX_DELTA = 0.01
    SY_DELTA = 0.01
    SXY_DELTA = 0.01

    def _test_at_points(self, python_func, fortran_stresses):
        for i, pnt in enumerate(self.TEST_POINTS):
            # compare x-dir stress
            self.assertAlmostEqual(
                python_func.stress(pnt[0], pnt[1])[0],
                fortran_stresses[0][0][i],
                delta=self.SX_DELTA
            )
            # compare y-dir stress
            self.assertAlmostEqual(
                python_func.stress(pnt[0], pnt[1])[1],
                fortran_stresses[1][0][i],
                delta=self.SY_DELTA
            )
            # compare shear stress
            self.assertAlmostEqual(
                python_func.stress(pnt[0], pnt[1])[2],
                fortran_stresses[2][0][i],
                delta=self.SXY_DELTA
            )

    def test_stresses_at_hole_boundary_with_only_Nx(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = np.array([[2.65646e-6, -8.91007e-7, 0.], [-8.91007e-7, 2.65646e-6, 0.], [0., 0., 7.09494e-6]])
        d = 0.25
        h = 0.058
        N = 100.
        beta = 0.
        p_stress = UnloadedHole([N, 0, 0], d, h, a_inv)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, 0, len(self.TEST_POINTS))
        self._test_at_points(p_stress, f_stress)

    def test_stresses_at_hole_boundary_with_only_px(self):
        from lekhnitskii import LoadedHole
        from tests.fortran import lekhnitskii_f as f_code
        a_inv = np.array([[2.65646e-6, -8.91007e-7, 0.], [-8.91007e-7, 2.65646e-6, 0.], [0., 0., 7.09494e-6]])
        d = 0.25
        h = 0.058
        p = 100.
        alpha = 345.
        p_stress = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        f_stress, f_u, f_v = f_code.loaded(4*p/h, d, a_inv, alpha, 0, len(self.TEST_POINTS))
        self._test_at_points(p_stress, f_stress)


if __name__ == '__main__':
    unittest.main()
