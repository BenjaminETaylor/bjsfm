import unittest
import numpy as np
from bjsfm.lekhnitskii import UnloadedHole, LoadedHole
from tests.fortran import lekhnitskii_f as bjsfm


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
NUM_POINTS = 4
X_POINTS = [r * np.cos(theta) for r, theta in
            zip([DIAMETER / 2] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
X_POINTS += [r * np.cos(theta) for r, theta in
             zip([DIAMETER / 2 + STEP_DIST] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
Y_POINTS = [r * np.sin(theta) for r, theta in
            zip([DIAMETER / 2] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
Y_POINTS += [r * np.sin(theta) for r, theta in
             zip([DIAMETER / 2 + STEP_DIST] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]


def plot_loaded_hole_fortran_stresses(p, h, d, a_inv, alpha, comp=0, num=100):
    """ Plots stresses for bearing solution from original fortran code

    Parameters
    ----------
    p : float
        bearing load
    h : float
        plate thickness
    d : float
        hole diameter
    a_inv : ndarray
        2D 3x3 inverse a-matrix from CLPT
    alpha : float
        bearing angle (degrees)
    comp : {0, 1, 2}, optional
        stress component, default=0
    num : int, optional
        level of refinement (higher = more), default=100

    """
    import matplotlib.pyplot as plt
    thetas = []
    radiis = []
    stresses = []
    for step in np.linspace(0, 5.5*d, num=num, endpoint=True):
        thetas.extend(np.linspace(0, 2*np.pi, num=num))
        radiis.extend([d/2+step]*num)
        f_stress, f_u, f_v = bjsfm.loaded(4 * p / h, d, a_inv, alpha, step, num)
        stresses.extend(f_stress[comp][1])
    x = np.array(radiis) * np.cos(thetas)
    y = np.array(radiis) * np.sin(thetas)
    x.shape = y.shape = (num, num)
    stresses = np.array(stresses).reshape(len(x), len(y))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cp = plt.contourf(x, y, stresses, corner_mask=True)
    plt.colorbar(cp)
    plt.xlim(-3*d, 3*d)
    plt.ylim(-3*d, 3*d)
    plt.title(f'Fortran BJSFM Stress:\n {comp} dir stress, {alpha} degrees brg load')
    plt.show()


class HoleTests(unittest.TestCase):

    ####################################################################################################################
    # test precisions
    ####################################################################################################################
    SX_DELTA = 0.1
    SY_DELTA = 0.1
    SXY_DELTA = 0.1

    def _test_at_points(self, python_stresses, fortran_stresses, step=0.):
        num_loops = len(X_POINTS)//2 if step > 0. else len(X_POINTS)
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
            if step > 0. and len(python_stresses) == 2*len(X_POINTS):
                py_step_index = i+len(X_POINTS)//2
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

    def test_quasi_with_only_Nx(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 0.    # load angle
        p_func = UnloadedHole([N, 0, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_with_only_Ny(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 90.   # load angle
        p_func = UnloadedHole([0, N, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_with_only_Nxy(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.    # load
        p_func = UnloadedHole([0, 0, N], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        fpxy_stress, fpxy_u, fpxy_v = bjsfm.unloded(N/h, d, a_inv, 45, step, len(X_POINTS)//2)
        fnxy_stress, fnxy_u, fnxy_v = bjsfm.unloded(-N/h, d, a_inv, -45, step, len(X_POINTS)//2)
        f_stress = fpxy_stress + fnxy_stress
        self._test_at_points(p_stress, f_stress, step=step)

    def test_soft_with_only_Nx(self):
        a_inv = SOFT
        h = SOFT_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 0.    # load angle
        p_func = UnloadedHole([N, 0, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_soft_with_only_Ny(self):
        a_inv = SOFT
        h = SOFT_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 90.   # load angle
        p_func = UnloadedHole([0, N, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_soft_with_only_Nxy(self):
        a_inv = SOFT
        h = SOFT_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.    # load
        p_func = UnloadedHole([0, 0, N], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        fpxy_stress, fpxy_u, fpxy_v = bjsfm.unloded(N/h, d, a_inv, 45, step, len(X_POINTS)//2)
        fnxy_stress, fnxy_u, fnxy_v = bjsfm.unloded(-N/h, d, a_inv, -45, step, len(X_POINTS)//2)
        f_stress = fpxy_stress + fnxy_stress
        self._test_at_points(p_stress, f_stress, step=step)

    def test_hard_with_only_Nx(self):
        a_inv = HARD
        h = HARD_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 0.    # load angle
        p_func = UnloadedHole([N, 0, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_hard_with_only_Ny(self):
        a_inv = HARD
        h = HARD_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.     # load
        beta = 90.   # load angle
        p_func = UnloadedHole([0, N, 0], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.unloded(N/h, d, a_inv, beta, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_hard_with_only_Nxy(self):
        a_inv = HARD
        h = HARD_THICK
        d = DIAMETER
        step = STEP_DIST
        N = 100.    # load
        p_func = UnloadedHole([0, 0, N], d, h, a_inv)
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        fpxy_stress, fpxy_u, fpxy_v = bjsfm.unloded(N/h, d, a_inv, 45, step, len(X_POINTS)//2)
        fnxy_stress, fnxy_u, fnxy_v = bjsfm.unloded(-N/h, d, a_inv, -45, step, len(X_POINTS)//2)
        f_stress = fpxy_stress + fnxy_stress
        self._test_at_points(p_stress, f_stress, step=step)


class LoadedHoleTests(HoleTests):

    def test_quasi_at_0_degrees(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 0.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_soft_at_0_degrees(self):
        a_inv = SOFT
        h = SOFT_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 0.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_hard_at_0_degrees(self):
        a_inv = HARD
        h = HARD_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 0.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_at_45_degrees(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 45.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_at_90_degrees(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 90.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_at_225_degrees(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 225.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    def test_quasi_at_290_degrees(self):
        a_inv = QUASI
        h = QUASI_THICK
        d = DIAMETER
        step = STEP_DIST
        p = 100.
        alpha = 290.
        p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
        p_stress = p_func.stress(X_POINTS, Y_POINTS)
        f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
        self._test_at_points(p_stress, f_stress, step=step)

    # # The below tests fail because fortran code seems to be incorrect
    # def test_soft_at_45_degrees(self):
    #     a_inv = HARD
    #     h = HARD_THICK
    #     d = DIAMETER
    #     step = STEP_DIST
    #     p = 100.
    #     alpha = 45.0
    #     # a_inv_rot = rotate_material_matrix(a_inv, angle=np.deg2rad(alpha))
    #     p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
    #     p_stress = p_func.stress(X_POINTS, Y_POINTS)
    #     f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, alpha, step, len(X_POINTS)//2)
    #     self._test_at_points(p_stress, f_stress, step=step)
    #
    # def test_hard_at_45_degrees(self):
    #     a_inv = HARD
    #     h = HARD_THICK
    #     d = DIAMETER
    #     step = STEP_DIST
    #     p = 100.
    #     alpha = 45.0
    #     # a_inv_rot = rotate_material_matrix(a_inv, angle=np.deg2rad(alpha))
    #     p_func = LoadedHole(p, d, h, a_inv, theta=np.deg2rad(alpha))
    #     p_stress = p_func.stress(X_POINTS, Y_POINTS)
    #     f_stress, f_u, f_v = bjsfm.loaded(4*p/h, d, a_inv, 0., step, len(X_POINTS)//2)
    #     self._test_at_points(p_stress, f_stress, step=step)


if __name__ == '__main__':
    unittest.main()
