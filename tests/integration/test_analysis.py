import unittest
from numpy.testing import assert_array_almost_equal
from bjsfm.lekhnitskii import rotate_plane_stress, LoadedHole, UnloadedHole
from bjsfm.analysis import MaxStrain
from tests.test_data import *


class TestMaxStrainQuasi(unittest.TestCase):

    def setUp(self):
        self.analysis = MaxStrain(
            QUASI, QUASI_THICK,
            DIAMETER,
            et0=QUASI_UNT, et90=QUASI_UNT, et45=QUASI_UNT, etn45=QUASI_UNT,
            ec0=QUASI_UNC, ec90=QUASI_UNC, ec45=QUASI_UNC, ecn45=QUASI_UNC,
            es0=QUASI_SBS, es90=QUASI_SBS, es45=QUASI_SBS, esn45=QUASI_SBS,
        )

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
        bypass_correction = rotate_plane_stress(np.array([p/(2*w), 0, 0]), angle=-theta)
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
        bypass = [100., 0., 0.]
        margins = self.analysis.analyze(bearing, bypass, rc=rc, num=num)
        e0, e90, es0 = self.analysis.strains(bearing, bypass, rc=rc, num=num)[1]
        s0, s90, ss0 = self.analysis.stresses(bearing, bypass, rc=rc, num=num)[1]
        calc_strains = QUASI_INV @ np.array([s0, s90, ss0])*QUASI_THICK
        self.assertAlmostEqual(e0, calc_strains[0])
        self.assertAlmostEqual(e90, calc_strains[1])
        self.assertAlmostEqual(es0, calc_strains[2])
        s45, sn45, ss45 = rotate_plane_stress(np.array([s0, s90, ss0]), angle=np.deg2rad(45.))
        e45, en45, es45 = QUASI_INV @ np.array([s45, sn45, ss45])*QUASI_THICK
        self.assertAlmostEqual(margins[1, 0], QUASI_UNT/e0 - 1)
        self.assertAlmostEqual(margins[1, 1], -QUASI_UNC/e90 - 1)
        self.assertAlmostEqual(margins[1, 2], QUASI_SBS/abs(es0) - 1)
        self.assertAlmostEqual(margins[1, 3], QUASI_UNT/e45 - 1)
        self.assertAlmostEqual(margins[1, 4], QUASI_UNT/en45 - 1)
        self.assertAlmostEqual(margins[1, 5], QUASI_SBS/abs(es45) - 1)


if __name__ == '__main__':
    unittest.main()

