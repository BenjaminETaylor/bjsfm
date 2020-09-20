import unittest
import numpy as np


class HoleTests(unittest.TestCase):

    # points to test must be equally spaced around hole, starting at zero degrees
    # laminate material matrix (inverse of CLPT A-matrix)
    QUASI = np.array(
        [[2.65646e-6, -8.91007e-7, 0.],
         [-8.91007e-7, 2.65646e-6, 0.],
         [0., 0., 7.09494e-6]]
    )
    THICKNESS = 0.058  # laminate thickness
    STEP_DIST = 0.15
    TEST_POINTS = ((0.125, 0.), (0., 0.125), (-0.125, 0.), (0., -0.125),
                   (0.125+STEP_DIST, 0.), (0., 0.125+STEP_DIST), (-0.125-STEP_DIST, 0.), (0., -0.125-STEP_DIST))
    SX_DELTA = 0.01
    SY_DELTA = 0.01
    SXY_DELTA = 0.01

    def _test_at_points(self, python_stresses, fortran_stresses, step=0.):
        num_loops = len(self.TEST_POINTS)//2 if step > 0. else len(self.TEST_POINTS)
        for i in range(num_loops):
            # compare x-dir stress at hole boundary
            self.assertAlmostEqual(
                python_stresses[i][0],
                fortran_stresses[0][0][i],
                delta=self.SX_DELTA
            )
            # compare y-dir stress at hole boundary
            self.assertAlmostEqual(
                python_stresses[i][1],
                fortran_stresses[1][0][i],
                delta=self.SY_DELTA
            )
            # compare shear stress at hole boundary
            self.assertAlmostEqual(
                python_stresses[i][2],
                fortran_stresses[2][0][i],
                delta=self.SXY_DELTA
            )
            if step > 0. and len(python_stresses) == 2*len(self.TEST_POINTS):
                py_step_index = i+len(self.TEST_POINTS)/2
                # compare x-dir stress at step distance
                self.assertAlmostEqual(
                    python_stresses[py_step_index][0],
                    fortran_stresses[0][1][i],
                    delta=self.SX_DELTA
                )
                # compare y-dir stress at step distance
                self.assertAlmostEqual(
                    python_stresses[py_step_index][1],
                    fortran_stresses[1][1][i],
                    delta=self.SY_DELTA
                )
                # compare shear stress at step distance
                self.assertAlmostEqual(
                    python_stresses[py_step_index][2],
                    fortran_stresses[2][1][i],
                    delta=self.SXY_DELTA
                )


class UnLoadedHoleTests(HoleTests):

    def test_stresses_at_with_only_Nx(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.THICKNESS
        step = self.STEP_DIST
        num_f_pnts = len(self.TEST_POINTS)/2 if step > 0. else len(self.TEST_POINTS)
        d = 0.25     # diameter
        N = 100.     # load
        beta = 0.    # load angle
        p_func = UnloadedHole([N, 0, 0], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, num_f_pnts)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_stresses_at_with_only_Ny(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.THICKNESS
        step = self.STEP_DIST
        num_f_pnts = len(self.TEST_POINTS)/2 if step > 0. else len(self.TEST_POINTS)
        d = 0.25     # diameter
        N = 100.     # load
        beta = 90.   # load angle
        p_func = UnloadedHole([0, N, 0], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, num_f_pnts)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_stresses_at_with_only_Nxy(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.THICKNESS
        step = self.STEP_DIST
        num_f_pnts = len(self.TEST_POINTS)/2 if step > 0. else len(self.TEST_POINTS)
        d = 0.25    # diameter
        N = 100.    # load
        p_func = UnloadedHole([0, 0, N], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        fpxy_stress, fpxy_u, fpxy_v = bjsfm.unloded(N/h, d, a_inv, 45, step, num_f_pnts)
        fnxy_stress, fnxy_u, fnxy_v = bjsfm.unloded(-N/h, d, a_inv, -45, step, num_f_pnts)
        f_stress = fpxy_stress + fnxy_stress
        self._test_at_points(p_stress, f_stress, step=step)


class LoadedHoleTests(HoleTests):

    def test_stresses_at_hole_boundary_with_only_px(self):
        from lekhnitskii import LoadedHole
        from tests.fortran import lekhnitskii_f as f_code
        a_inv = self.QUASI
        h = self.THICKNESS
        step = self.STEP_DIST
        num_f_pnts = len(self.TEST_POINTS)/2 if step > 0. else len(self.TEST_POINTS)
        d = 0.25
        p = 100.
        alpha = 0.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(self.TEST_POINTS)
        f_stress, f_u, f_v = f_code.loaded(4*p/h, d, a_inv, alpha, step, num_f_pnts)
        self._test_at_points(p_stress, f_stress, step=step)


if __name__ == '__main__':
    unittest.main()
