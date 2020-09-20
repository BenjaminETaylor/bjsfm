import unittest
import numpy as np


class HoleTests(unittest.TestCase):

    ####################################################################################################################
    # geometry/material inputs
    ####################################################################################################################
    # laminate material matrix (inverse of CLPT A-matrix)
    # Hexcel 8552 IM7 Unidirectional
    # ref: https://www.wichita.edu/research/NIAR/Documents/Qual-CAM-RP-2009-015-Rev-B-Hexcel-8552-IM7-MPDR-04.16.19.pdf
    # E1c[RTD] = 20.04 Msi
    # E2c[RTD] = 1.41 Msi
    # nu12[RTD] = 0.356
    # nu21[RTD] = 0.024
    # G12[RTD] = 0.68 Msi
    # CPT = 0.0072 in
    # QUASI [25/50/25], HARD [70/20/10], SOFT [10/80/10]
    QUASI = np.linalg.inv(np.array(
        [[988374.5, 316116.9, 0.],
         [316116.9, 988374.5, 0.],
         [0., 0., 336128.8]]
    ))
    HARD = np.linalg.inv(np.array(
        [[2240564.3, 201801.5, 0.],
         [201801.5, 617061.1, 0.],
         [0., 0., 226816.4]]
    ))
    SOFT = np.linalg.inv(np.array(
        [[1042123.5, 588490.7, 0.],
         [588490.7, 1042123.5, 0.],
         [0., 0., 613505.6]]
    ))
    QUASI_THICK = 0.1152  # laminate thickness
    HARD_THICK = SOFT_THICK = 0.144
    DIAMETER = 0.25
    ####################################################################################################################
    # test point inputs
    ####################################################################################################################
    # points to test must be equally spaced around hole, starting at zero degrees
    # there must be two rows of points; one at the hole boundary, and another at step distance
    STEP_DIST = 0.15
    NUM_POINTS = 6
    TEST_POINTS = [(r*np.cos(theta), r*np.sin(theta))
                   for r, theta in
                   zip([DIAMETER/2]*NUM_POINTS, np.linspace(0, 2*np.pi, num=NUM_POINTS, endpoint=False))]
    TEST_POINTS += [(r*np.cos(theta), r*np.sin(theta))
                    for r, theta in
                    zip([DIAMETER/2+STEP_DIST]*NUM_POINTS, np.linspace(0, 2*np.pi, num=NUM_POINTS, endpoint=False))]
    ####################################################################################################################
    # test precisions
    ####################################################################################################################
    SX_DELTA = 0.1
    SY_DELTA = 0.1
    SXY_DELTA = 0.1

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

    def test_stresses_with_only_Nx(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.QUASI_THICK
        d = self.DIAMETER
        step = self.STEP_DIST
        N = 100.     # load
        beta = 0.    # load angle
        p_func = UnloadedHole([N, 0, 0], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(self.TEST_POINTS)/2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_stresses_with_only_Ny(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.QUASI_THICK
        d = self.DIAMETER
        step = self.STEP_DIST
        N = 100.     # load
        beta = 90.   # load angle
        p_func = UnloadedHole([0, N, 0], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(self.TEST_POINTS)/2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_stresses_with_only_Nxy(self):
        from lekhnitskii import UnloadedHole
        from tests.fortran import lekhnitskii_f as bjsfm
        a_inv = self.QUASI
        h = self.QUASI_THICK
        d = self.DIAMETER
        step = self.STEP_DIST
        N = 100.    # load
        p_func = UnloadedHole([0, 0, N], d, h, a_inv)
        p_stress = p_func.stress(self.TEST_POINTS)
        fpxy_stress, fpxy_u, fpxy_v = bjsfm.unloded(N/h, d, a_inv, 45, step, len(self.TEST_POINTS)/2)
        fnxy_stress, fnxy_u, fnxy_v = bjsfm.unloded(-N/h, d, a_inv, -45, step, len(self.TEST_POINTS)/2)
        f_stress = fpxy_stress + fnxy_stress
        self._test_at_points(p_stress, f_stress, step=step)


class LoadedHoleTests(HoleTests):

    def test_stresses_for_multiple_bearing_angles(self):
        from lekhnitskii import LoadedHole
        from tests.fortran import lekhnitskii_f as f_code
        a_inv = self.QUASI
        h = self.QUASI_THICK
        d = self.DIAMETER
        step = self.STEP_DIST
        p = 100.
        # test bearing load at multiple angles
        for alpha in np.linspace(0, 360, 20, endpoint=False):
            p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
            p_stress = p_func.stress(self.TEST_POINTS)
            f_stress, f_u, f_v = f_code.loaded(4*p/h, d, a_inv, alpha, step, len(self.TEST_POINTS)/2)
            self._test_at_points(p_stress, f_stress, step=step)


if __name__ == '__main__':
    unittest.main()
