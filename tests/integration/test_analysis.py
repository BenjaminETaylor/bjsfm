import unittest

import matplotlib
matplotlib.use('Agg')  # headless backend so plotting smoke-tests never try to open a window
import matplotlib.pyplot as plt

from numpy.testing import assert_array_almost_equal
from bjsfm.lekhnitskii import rotate_stress, rotate_strain, LoadedHole, UnloadedHole
from bjsfm.analysis import Analysis, MaxStrain
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
        self.assertAlmostEqual(analysis.et[0], QUASI_UNT)
        self.assertAlmostEqual(analysis.et[30], QUASI_UNT)
        self.assertAlmostEqual(analysis.ec[90], QUASI_UNC)
        self.assertAlmostEqual(analysis.ec[60], QUASI_UNC)
        self.assertAlmostEqual(analysis.es[45], SHEAR_STRN)
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
            for iangle, angle in enumerate(self.analysis.angles):
                rotated_strains = rotate_strain(strains, angle=np.deg2rad(angle))
                # axial strains
                x_strains = rotated_strains[:, 0]
                compare_margins[:, iangle*2] = np.select(
                    [x_strains > 0, x_strains < 0], [QUASI_UNT / x_strains - 1, -abs(QUASI_UNC) / x_strains - 1])
                # shear strains
                xy_strains = np.abs(rotated_strains[:, 2])
                compare_margins[:, iangle*2+1] = abs(SHEAR_STRN) / xy_strains - 1
        # margins for near-zero strains legitimately blow up to very large (or inf) values, so a
        # relative tolerance is required; an absolute one (assert_array_almost_equal) is meaningless
        # at magnitudes of ~1e17.
        np.testing.assert_allclose(margins, compare_margins, rtol=1e-6)

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

    def test_polar_points(self):
        rc = 0.1
        num = 8
        r, theta = self.analysis.polar_points(rc=rc, num=num)
        self.assertEqual(len(r), num)
        self.assertEqual(len(theta), num)
        assert_array_almost_equal(r, np.full(num, DIAMETER/2 + rc))
        assert_array_almost_equal(theta, np.linspace(0, 2*np.pi, num=num, endpoint=False))

    def test_displacements_superposition(self):
        # displacements must equal the superposition of the loaded + unloaded hole solutions
        # (NaNs from @_remove_bad_displacments occur in both and compare equal)
        bearing = [100., 0.]
        bypass = [50., 0., 0.]
        rc = 0.25
        num = 50
        p, theta = self.analysis.bearing_angle(bearing)
        brg = LoadedHole(p, DIAMETER, QUASI_THICK, QUASI_INV, theta=theta)
        byp = UnloadedHole(bypass, DIAMETER, QUASI_THICK, QUASI_INV)
        x, y = self.analysis.xy_points(rc=rc, num=num)
        expected = byp.displacement(x, y) + brg.displacement(x, y)
        result = self.analysis.displacements(bearing, bypass, rc=rc, num=num)
        self.assertEqual(result.shape, (num, 2))
        assert_array_almost_equal(result, expected)

    def test_unloaded_width_correction_sign(self):
        # a nonzero bypass in the bearing direction should set the sign of the DeJong correction
        w = 6*DIAMETER
        bearing = [100., 0.]
        rc = 0.25
        num = 50
        for nx_sign in (1., -1.):
            bypass = [nx_sign*200., 0., 0.]
            p, theta = self.analysis.bearing_angle(bearing)
            corrected = np.array(bypass, dtype=float)
            corrected += rotate_stress(np.array([p/(2*w)*nx_sign, 0., 0.]), angle=-theta)
            byp = UnloadedHole(corrected, DIAMETER, QUASI_THICK, QUASI_INV)
            brg = LoadedHole(p, DIAMETER, QUASI_THICK, QUASI_INV, theta=theta)
            x, y = self.analysis.xy_points(rc=rc, num=num)
            expected = byp.stress(x, y) + brg.stress(x, y)
            result = self.analysis.stresses(bearing, bypass, rc=rc, num=num, w=w)
            # stresses reach ~1e3, so a relative tolerance is appropriate
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)

    def test_bearing_front_mask_no_exclusion(self):
        # falsy exclusion angle -> keep everything
        mask = self.analysis.bearing_front_mask([100., 0.], num=20, exclusion_angle=0.)
        self.assertTrue(np.all(mask))

    def test_bearing_front_mask_no_bearing(self):
        # zero bearing load -> no front to exclude, keep everything
        mask = self.analysis.bearing_front_mask([0., 0.], num=20, exclusion_angle=30.)
        self.assertTrue(np.all(mask))

    def test_bearing_front_mask_excludes_front(self):
        num = 36  # 10-degree spacing, theta[0] == 0 (the loaded front for +x bearing)
        exclusion_angle = 30.
        mask = self.analysis.bearing_front_mask([100., 0.], num=num, exclusion_angle=exclusion_angle)
        _, theta = self.analysis.polar_points(num=num)
        ang_dist = np.abs(np.angle(np.exp(1j*theta)))  # distance from bearing direction (0 deg)
        expected = ang_dist > np.deg2rad(exclusion_angle)
        assert_array_almost_equal(mask, expected)
        self.assertFalse(mask[0])  # point directly in front of the load is excluded
        self.assertTrue(mask[num//2])  # point directly behind the load is kept

    def test_analyze_exclusion_angle(self):
        # excluded (front) points are reported with infinite margin and never govern
        rc = 0.15
        num = 36
        bearing = [1000., 0.]
        bypass = [0., 0., 0.]
        exclusion_angle = 30.
        margins = self.analysis.analyze(bearing, bypass, rc=rc, num=num, exclusion_angle=exclusion_angle)
        keep = self.analysis.bearing_front_mask(bearing, num=num, exclusion_angle=exclusion_angle)
        self.assertTrue(np.all(np.isinf(margins[~keep])))
        # kept rows should match a plain analysis (no exclusion) at those same rows
        plain = self.analysis.analyze(bearing, bypass, rc=rc, num=num)
        # margins range over several orders of magnitude, so use a relative tolerance
        np.testing.assert_allclose(margins[keep], plain[keep], rtol=1e-6)


class TestBearingBypassCurve(unittest.TestCase):

    def setUp(self):
        self.analysis = MaxStrain(
            QUASI, QUASI_THICK,
            DIAMETER,
            et={0: QUASI_UNT, 90: QUASI_UNT, 45: QUASI_UNT, -45: QUASI_UNT},
            ec={0: QUASI_UNC, 90: QUASI_UNC, 45: QUASI_UNC, -45: QUASI_UNC},
            es={0: SHEAR_STRN, 90: SHEAR_STRN, 45: SHEAR_STRN, -45: SHEAR_STRN},
        )
        self.rc = 0.05
        self.num = 50
        self.d_t = DIAMETER * QUASI_THICK
        self.a00 = QUASI_INV[0, 0]

    def test_envelope_shape_and_signs(self):
        npoints = 40
        brg_stress, byp_strain = self.analysis.bearing_bypass_curve(
            npoints=npoints, rc=self.rc, num=self.num)
        # npoints sampled points plus a closing point on the bearing-stress axis
        self.assertEqual(len(brg_stress), npoints + 1)
        self.assertEqual(len(byp_strain), npoints + 1)
        # bearing stress sweeps from zero and is non-negative; bypass strain is non-negative
        self.assertAlmostEqual(brg_stress[0], 0.)
        self.assertTrue(np.all(brg_stress >= 0.))
        self.assertTrue(np.all(byp_strain >= 0.))
        # rising portion is monotonically increasing in bearing stress
        self.assertTrue(np.all(np.diff(brg_stress[:npoints]) > 0))
        # the curve is closed down to the axis at the right end
        self.assertAlmostEqual(brg_stress[-1], brg_stress[-2])
        self.assertAlmostEqual(byp_strain[-1], 0.)
        # bypass capacity decreases as bearing increases (collinear loads compete)
        self.assertGreater(byp_strain[0], byp_strain[npoints - 1])

    def test_envelope_points_are_critical(self):
        # every (bearing stress, bypass strain) point on the envelope is by construction the
        # largest admissible bypass strain, so the governing margin of safety there is ~0
        npoints = 25
        brg_stress, byp_strain = self.analysis.bearing_bypass_curve(
            npoints=npoints, rc=self.rc, num=self.num)
        for s, e in zip(brg_stress[:npoints], byp_strain[:npoints]):
            if e <= 0:
                continue
            bearing = [s*self.d_t, 0.]
            bypass = [e/self.a00, 0., 0.]
            margins = self.analysis.analyze(bearing, bypass, rc=self.rc, num=self.num)
            self.assertAlmostEqual(np.min(margins), 0., places=5)

    def test_superposition_consistency(self):
        # the combined strain field is the linear superposition of the unit bearing and unit bypass
        # fields used internally by bearing_bypass_curve
        s, e = 2000., 0.003
        brg_field = self.analysis.strains([self.d_t, 0.], [0., 0., 0.], rc=self.rc, num=self.num)
        byp_field = self.analysis.strains([0., 0.], [1./self.a00, 0., 0.], rc=self.rc, num=self.num)
        combined = self.analysis.strains([s*self.d_t, 0.], [e/self.a00, 0., 0.], rc=self.rc, num=self.num)
        assert_array_almost_equal(combined, brg_field*s + byp_field*e)

    def test_brg_allow_caps_envelope(self):
        # establish the natural failure bearing stress with no cutoff
        brg_stress, _ = self.analysis.bearing_bypass_curve(npoints=30, rc=self.rc, num=self.num)
        s0 = brg_stress[-2]
        # a smaller allowable should cut the sweep off early
        allow = s0/2
        capped, _ = self.analysis.bearing_bypass_curve(
            brg_allow=allow, npoints=30, rc=self.rc, num=self.num)
        self.assertAlmostEqual(capped[-2]/allow, 1., places=6)
        # an allowable larger than s0 leaves the (strain-limited) envelope unchanged
        loose, _ = self.analysis.bearing_bypass_curve(
            brg_allow=s0*2, npoints=30, rc=self.rc, num=self.num)
        self.assertAlmostEqual(loose[-2]/s0, 1., places=6)

    def test_no_allowables_returns_empty(self):
        analysis = MaxStrain(QUASI, QUASI_THICK, DIAMETER)  # no et/ec/es supplied
        brg_stress, byp_strain = analysis.bearing_bypass_curve(rc=self.rc, num=self.num)
        self.assertEqual(len(brg_stress), 0)
        self.assertEqual(len(byp_strain), 0)


class TestStrainMargins(unittest.TestCase):

    def test_shear_only(self):
        strains = np.array([[0.001, 0., 0.0005], [-0.002, 0., -0.001]])
        margins = MaxStrain._strain_margins(strains, es=SHEAR_STRN)
        # no et/ec -> normal-strain margin is infinite
        self.assertTrue(np.all(np.isinf(margins[:, 0])))
        assert_array_almost_equal(margins[:, 1], abs(SHEAR_STRN)/np.abs(strains[:, 2]) - 1)

    def test_tension_only_skips_compression(self):
        strains = np.array([[0.001, 0., 0.], [-0.001, 0., 0.]])
        margins = MaxStrain._strain_margins(strains, et=QUASI_UNT)
        # tension allowable only governs the positive normal strain; the negative one stays infinite
        self.assertAlmostEqual(margins[0, 0], QUASI_UNT/0.001 - 1)
        self.assertTrue(np.isinf(margins[1, 0]))
        # no shear allowable -> shear margin infinite
        self.assertTrue(np.all(np.isinf(margins[:, 1])))

    def test_compression_only_skips_tension(self):
        strains = np.array([[0.001, 0., 0.], [-0.001, 0., 0.]])
        margins = MaxStrain._strain_margins(strains, ec=QUASI_UNC)
        self.assertTrue(np.isinf(margins[0, 0]))
        self.assertAlmostEqual(margins[1, 0], -abs(QUASI_UNC)/-0.001 - 1)


class TestPlottingSmoke(unittest.TestCase):

    def setUp(self):
        self.analysis = MaxStrain(
            QUASI, QUASI_THICK, DIAMETER,
            et={0: QUASI_UNT, 90: QUASI_UNT, 45: QUASI_UNT, -45: QUASI_UNT},
            ec={0: QUASI_UNC, 90: QUASI_UNC, 45: QUASI_UNC, -45: QUASI_UNC},
            es={0: SHEAR_STRN, 90: SHEAR_STRN, 45: SHEAR_STRN, -45: SHEAR_STRN},
        )

    def tearDown(self):
        plt.close('all')

    def test_plot_stress_runs(self):
        fig, ax = plt.subplots()
        self.analysis.plot_stress([100., 0.], [50., 0., 0.], comp='x', rnum=20, tnum=20, axes=ax)

    def test_plot_displacement_runs(self):
        fig, ax = plt.subplots()
        self.analysis.plot_displacement([100., 0.], [50., 0., 0.], comp='x', rnum=20, tnum=20, axes=ax)

    def test_plot_bearing_bypass_runs(self):
        fig, ax = plt.subplots()
        self.analysis.plot_bearing_bypass(npoints=20, rc=0.05, num=20, axes=ax, label='quasi')


if __name__ == '__main__':
    unittest.main()





















