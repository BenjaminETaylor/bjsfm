import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from bjsfm.lekhnitskii import rotate_stress, rotate_strain, LoadedHole, UnloadedHole
from bjsfm.analysis import MaxStrain
from tests.test_data import *


class TestMaxStrainQuasi(unittest.TestCase):

    def setUp(self):
        self.analysis = MaxStrain(
            QUASI, QUASI_THICK,
            DIAMETER,
            et={0: QUASI_UNT, 90: QUASI_UNT, 45: QUASI_UNT, -45: QUASI_UNT},
            ec={0: QUASI_UNC, 90: QUASI_UNC, 45: QUASI_UNC, -45: QUASI_UNC},
            es={0: SHEAR_STRN, 90: SHEAR_STRN, 45: SHEAR_STRN, -45: SHEAR_STRN},
        )
    
    def test_equalize_dicts(self):
        analysis = MaxStrain(
            QUASI, QUASI_THICK,
            DIAMETER,
            et={0: QUASI_UNT, 30: QUASI_UNT},
            ec={90: QUASI_UNC, 60: QUASI_UNC},
            es={45: SHEAR_STRN},
        )
        self.assertTrue(analysis.et.keys() == analysis.ec.keys() == analysis.es.keys())
        self.assertAlmostEquals(analysis.et[0], QUASI_UNT)
        self.assertAlmostEquals(analysis.et[30], QUASI_UNT)
        self.assertAlmostEquals(analysis.ec[90], QUASI_UNC)
        self.assertAlmostEquals(analysis.ec[60], QUASI_UNC)
        self.assertAlmostEquals(analysis.es[45], SHEAR_STRN)
        self.assertEqual(analysis.et[90], np.inf)

    def test_xy_points(self):
        rc = 0.
        num = 4
        x, y = self.analysis.xy_points(rc=rc, num=num)
        radius = DIAMETER/2
        self.assertAlmostEqual(x[0], radius)
        self.assertAlmostEqual(x[1], 0.)
        self.assertAlmostEqual(x[2], -radius)
        self.assertAlmostEqual(x[3], 0.)
        self.assertAlmostEqual(y[0], 0.)
        self.assertAlmostEqual(y[1], radius)
        self.assertAlmostEqual(y[2], 0.)
        self.assertAlmostEqual(y[3], -radius)

    def test_strain_rotation(self):
        bearing = [0, 0]
        bypass = [100, 0, 0]
        angles = map(np.deg2rad, [0, 45, 90, -45, 60, -60])
        for angle in angles:
            rotated_stresses = rotate_stress(self.analysis.stresses(bearing, bypass), angle=angle)
            assert_array_almost_equal(
                (self.analysis.a_inv @ (rotated_stresses * self.analysis.t).T).T,
                rotate_strain(self.analysis.strains(bearing, bypass), angle=angle),
            )

    def test_0_bearing_0_angle(self):
        bearing = [0, 0]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, 0.)
        self.assertAlmostEqual(np.rad2deg(theta), 0.)

    def test_bearing_0_angle(self):
        bearing = [100, 0]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, 100.)
        self.assertAlmostEqual(np.rad2deg(theta), 0.)

    def test_bearing_90_angle(self):
        bearing = [0, 100]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, 100.)
        self.assertAlmostEqual(np.rad2deg(theta), 90.)

    def test_bearing_180_angle(self):
        bearing = [-100, 0]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, 100.)
        self.assertAlmostEqual(np.rad2deg(theta), 180.)

    def test_bearing_n90_angle(self):
        bearing = [0, -100]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, 100.)
        self.assertAlmostEqual(np.rad2deg(theta), -90.)

    def test_bearing_45_angle(self):
        bearing = [100, 100]
        p, theta = self.analysis.bearing_angle(bearing)
        self.assertAlmostEqual(p, np.sqrt(np.sum(np.square(bearing))))
        self.assertAlmostEqual(np.rad2deg(theta), 45.)

    def test_width_0_angle(self):
        w = 6*DIAMETER
        bearing = [100, 0]
        bypass = [0, 0, 0]
        bypass_correction = [100/(2*w), 0, 0]
        brg = LoadedHole(bearing[0], DIAMETER, QUASI_THICK, QUASI_INV, theta=0.)
        byp = UnloadedHole(bypass_correction, DIAMETER, QUASI_THICK, QUASI_INV)
        x, y = self.analysis.xy_points(rc=0., num=100)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        total_stress = byp_stress + brg_stress
        analysis_stress = self.analysis.stresses(bearing, bypass, rc=0., num=100, w=w)
        assert_array_almost_equal(total_stress, analysis_stress)

    def test_width_45_angle(self):
        w = 6*DIAMETER
        bearing = [100, 100]
        bypass = [0, 0, 0]
        p, theta = self.analysis.bearing_angle(bearing)
        bypass_correction = rotate_stress(np.array([p/(2*w), 0, 0]), angle=-theta)
        brg = LoadedHole(p, DIAMETER, QUASI_THICK, QUASI_INV, theta=theta)
        byp = UnloadedHole(bypass_correction, DIAMETER, QUASI_THICK, QUASI_INV)
        x, y = self.analysis.xy_points(rc=0., num=100)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        total_stress = byp_stress + brg_stress
        analysis_stress = self.analysis.stresses(bearing, bypass, rc=0., num=100, w=w)
        assert_array_almost_equal(total_stress, analysis_stress)

    def test_max_strain(self):
        rc = 0.15
        num = 4
        bearing = [0, 0]
        bypass = [1000., 0., 0.]
        margins = self.analysis.analyze(bearing, bypass, rc=rc, num=num)
        strains = self.analysis.strains(bearing, bypass, rc=rc, num=num)
        compare_margins = np.empty((num, 2*len(self.analysis.et)))
        with np.errstate(divide='ignore'):
            for iangle, angle in enumerate(self.analysis.et):
                rotated_strains = rotate_strain(strains, angle=np.deg2rad(angle))
                # axial strains
                x_strains = rotated_strains[:, 0]
                compare_margins[:, iangle*2] = np.select(
                    [x_strains > 0, x_strains < 0], [QUASI_UNT / x_strains - 1, -abs(QUASI_UNC) / x_strains - 1])
                # shear strains
                xy_strains = np.abs(rotated_strains[:, 2])
                compare_margins[:, iangle*2+1] = abs(SHEAR_STRN) / xy_strains - 1
        assert_array_almost_equal(margins, compare_margins)

    def test_random_allowables(self):
        rc = 0.15
        num = 4
        bearing = [0, 0]
        bypass = [1000., 0., 0.]
        analysis = MaxStrain(
            QUASI, QUASI_THICK, DIAMETER,
            et={0: QUASI_UNT}, ec={90: QUASI_UNC}
        )
        margins = analysis.analyze(bearing, bypass, rc=rc, num=num)
        strains = analysis.strains(bearing, bypass, rc=rc, num=num)
        compare_margins = np.empty((num, 4))
        with np.errstate(divide='ignore'):
            # axial strains
            x_strains = strains[:, 0]
            compare_margins[:, 0] = np.select(
                [x_strains > 0, x_strains < 0], [QUASI_UNT / x_strains - 1, -np.inf / x_strains - 1])
            # shear strains
            xy_strains = np.abs(strains[:, 2])
            compare_margins[:, 1] = np.inf / xy_strains - 1
            rotated_strains = rotate_strain(strains, angle=np.deg2rad(90))
            # axial strains
            x_strains = rotated_strains[:, 0]
            compare_margins[:, 2] = np.select(
                [x_strains > 0, x_strains < 0], [np.inf / x_strains - 1, -abs(QUASI_UNC) / x_strains - 1])
            # shear strains
            xy_strains = np.abs(rotated_strains[:, 2])
            compare_margins[:, 3] = np.inf / xy_strains - 1
        assert_array_almost_equal(margins, compare_margins)

if __name__ == '__main__':
    unittest.main()





















