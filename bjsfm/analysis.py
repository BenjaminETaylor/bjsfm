import numpy as np
import bjsfm.lekhnitskii as lk


class MaxStrain:
    """A class for analyzing joint failure using max strain failure theory.

    Parameters
    ----------
    thickness : float
        plate thickness
    diameter : float
        hole diameter
    a_matrix : array_like
        2D 3x3 inverse A-matrix from CLPT
    et0 : float
        tension strain allowable in 0 deg direction
    et90 : float
        tension strain allowable in 90 deg direction
    et45 : float
        tension strain allowable in 45 deg direction
    etn45 : float
        tension strain allowable in -45 deg direction
    ec0 : float
        compression strain allowable in 0 deg direction
    ec90 : float
        compression strain allowable in 90 deg direction
    ec45 : float
        compression strain allowable in 45 deg direction
    ecn45 : float
        compression strain allowable in -45 deg direction
    es0 : float
        shear strain allowable in 0 deg direction
    es90 : float
        shear strain allowable in 90 deg direction
    es45 : float
        shear strain allowable in 45 deg direction
    esn45 : float
        shear strain allowable in -45 deg direction

    Attributes
    ----------
    t : float
        plate thickness
    r : float
        hole radius
    a : ndarray
        2D 3x3 A-matrix from CLPT
    a_inv : ndarray
        2D 3x3 Inverse A-matrix from CLPT
    et0 : float
        plate tension strain allowable in 0 deg direction
    et90 : float
        plate tension strain allowable in 90 deg direction
    et45 : float
        plate tension strain allowable in 45 deg direction
    etn45 : float
        plate tension strain allowable in -45 deg direction
    ec0 : float
        plate compression strain allowable in 0 deg direction
    ec90 : float
        plate compression strain allowable in 90 deg direction
    ec45 : float
        plate compression strain allowable in 45 deg direction
    ecn45 : float
        plate compression strain allowable in -45 deg direction
    es0 : float
        plate shear strain allowable in 0 deg direction
    es90 : float
        plate shear strain allowable in 90 deg direction
    es45 : float
        plate shear strain allowable in 45 deg direction
    esn45 : float
        plate shear strain allowable in -45 deg direction

    """

    def __init__(self, a_matrix, thickness, diameter, et0=None, et90=None, et45=None, etn45=None, ec0=None, ec90=None,
                 ec45=None, ecn45=None, es0=None, es90=None, es45=None, esn45=None):
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix)
        self.a_inv = np.linalg.inv(self.a)
        self.e_allow = {'et0': et0, 'et90': et90, 'et45': et45, 'etn45': etn45, 'ec0': ec0, 'ec90': ec90,
                        'ec45': ec45, 'ecn45': ecn45, 'es0': es0, 'es90': es90, 'es45': es45, 'esn45': esn45}

    def _radial_points(self, step, num_pnts):
        """Calculates x, y points

        Parameters
        ----------
        step : float
            distance away from hole
        num_pnts: int
            number of points

        Returns
        -------
        x, y : ndarray
            1D arrays, cartesian x, y locations

        """
        r = np.array([self.r + step] * num_pnts)
        theta = np.linspace(0, 2*np.pi, num=num_pnts, endpoint=False)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    @staticmethod
    def _strain_margins(strains, et0=None, ec0=None, et90=None, ec90=None, es0=None, es90=None):
        r"""Calculates margins of safety

        Parameters
        ----------
        strains : ndarray
            2D nx3 array of [[:math: `\epsilon_x, \epsilon_y, \gamma_{xy}`], ...] in-plane strains
        et0 : float, optional
            tension strain allowable in 0 deg direction
        et90 : float, optional
            tension strain allowable in 90 deg direction
        ec0 : float, optional
            compression strain allowable in 0 deg direction
        ec90 : float, optional
            compression strain allowable in 90 deg direction
        es0 : float, optional
            shear strain allowable in 0 deg direction
        es90 : float, optional
            shear strain allowable in 90 deg direction

        Returns
        -------
        margins : ndarray
            2D nx3 array [[x-dir margin, y-dir margin, xy margin], ...]

        """
        margins = np.empty((strains.shape[0], 3))
        margins[:] = np.inf
        with np.errstate(divide='ignore'):
            # 0 deg direction
            if et0 and ec0:
                x_strains = strains[:, 0]
                margins[:, 0] = np.select(
                    [x_strains > 0, x_strains < 0], [et0/x_strains - 1, -abs(ec0)/x_strains - 1])
            # 90 deg direction
            if et90 and ec90:
                y_strains = strains[:, 1]
                margins[:, 1] = np.select(
                    [y_strains > 0, y_strains < 0], [et90/y_strains - 1, -abs(ec90)/y_strains - 1])
            # 0/90 shear
            if es0 and es90:
                xy_strains = np.abs(strains[:, 2])
                # i_nz = np.nonzero(xy_strains)
                es = min(abs(es0), abs(es90))
                margins[:, 2] = es/xy_strains - 1
        return margins

    @staticmethod
    def _rotate_strains(strains, angle=0.):
        r"""Rotates the strain components by given angle

        The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

        Parameters
        ----------
        strains : ndarray
            2D nx3 array of [:math: `\epsilon_x, \epsilon_y, \gamma_{xy}`] in-plane strains
        angle : float, default 0.
            angle measured counter-clockwise from positive x-axis (radians)

        Returns
        -------
        ndarray
            2D nx3 array of [:math: `\epsilon_x', \epsilon_y', \gamma_{xy}'`] rotated stresses

        """
        c = np.cos(angle)
        s = np.sin(angle)
        rotation_matrix = np.array([
            [c**2, s**2, 2*s*c],
            [s**2, c**2, -2*s*c],
            [-2*s*c, 2*s*c, c**2 - s**2]
        ])
        strains = rotation_matrix @ strains.T
        return strains.T

    def stresses(self, bearing, bypass, rc=0., num=100):
        """ Calculate stresses

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole

        Returns
        -------
        ndarray
            2D numx3 array of plate stresses

        """
        d = self.r*2
        t = self.t
        a_inv = self.a_inv
        alpha = np.tan(bearing[1]/bearing[0]) if abs(bearing[0]) > 0. else 0.
        p = np.sqrt(np.sum(np.square(bearing)))
        x, y = self._radial_points(rc, num)
        brg = lk.LoadedHole(p, d, t, a_inv, theta=alpha)
        byp = lk.UnloadedHole(bypass, d, t, a_inv)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        return byp_stress + brg_stress

    def strains(self, bearing, bypass, rc=0., num=100):
        """ Calculate strains

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole

        Returns
        -------
        ndarray
            2D numx3 array of plate strains

        """
        stresses = self.stresses(bearing, bypass, rc=rc, num=num)
        strains = self.a_inv @ stresses.T/self.t
        return strains.T

    def analyze(self, bearing, bypass, rc=0., num=100):
        """ Analyze the joint for a set of loads

        Parameters
        ----------
        bearing : array_like
            1D 1x2 array bearing load [Px, Py] (force)
        bypass : array_like
            1D 1x3 array bypass loads [Nx, Ny, Nxy] (force/unit-length)
        rc : float, default 0.
            characteristic distance
        num : int, default 100
            number of points to check around hole

        Returns
        -------
        ndarray
            2D numx6 array of margins of safety
            [[<0 deg margin>, <90 deg margin>, <0/90 shear margin>,
            <45 deg margin>, <-45 deg margin>, <45/-45 shear margin>], ...]

        """
        e_all = self.e_allow
        margins = np.empty((num, 6))
        # check 0/90 direction
        strains = self.strains(bearing, bypass, rc=rc, num=num)
        margins[:, :3] = self._strain_margins(strains, et0=e_all['et0'], ec0=e_all['ec0'], et90=e_all['et90'],
                                              ec90=e_all['ec90'], es0=e_all['es0'], es90=e_all['es90'])
        # check 45/-45 direction
        strains = self._rotate_strains(strains, angle=np.deg2rad(45))
        margins[:, 3:] = self._strain_margins(strains, et0=e_all['et45'], ec0=e_all['ec45'], et90=e_all['etn45'],
                                              ec90=e_all['ecn45'], es0=e_all['es45'], es90=e_all['esn45'])
        return margins


