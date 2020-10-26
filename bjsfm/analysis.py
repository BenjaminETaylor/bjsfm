import numpy as np
import bjsfm.lekhnitskii as lek


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

    def __init__(self, thickness, diameter,  a_matrix, et0=None, et90=None, et45=None, etn45=None, ec0=None, ec90=None,
                 ec45=None, ecn45=None, es0=None, es90=None, es45=None, esn45=None):
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix)
        self.a_inv = np.linalg.inv(self.a)
        self.e_allow = {'et0': et0, 'et90': et90, 'et45': et45, 'etn45': etn45, 'ec0': ec0, 'ec90': ec90,
                        'ec45': ec45, 'ecn45': ecn45, 'es0': es0, 'es90': es90, 'es45': es45, 'esn45': esn45}

    def _radial_points(self, step, num_pnts):
        """Calculates x, y points"""
        r = np.array([self.r + step] * num_pnts)
        theta = np.linspace(0, 2*np.pi, num=num_pnts, endpoint=False)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def _strain_margins(self, strains):
        """Calculates margins of safety"""
        e_allow = self.e_allow
        margins = np.empty((strains.shape[0], 6))
        margins[:] = np.nan
        # 0 deg direction
        if e_allow['et0'] and e_allow['ec0']:
            x_strains = strains[:, 0]
            margins[:, 0] = np.select(
                [x_strains > 0, x_strains < 0], [e_allow['et0']/x_strains - 1, -abs(e_allow['ec0'])/x_strains - 1])
        # 90 deg direction
        if e_allow['et90'] and e_allow['ec90']:
            y_strains = strains[:, 1]
            margins[:, 1] = np.select(
                [y_strains > 0, y_strains < 0], [e_allow['et90']/y_strains - 1, -abs(e_allow['ec90'])/y_strains - 1])
        # 0/90 shear
        if e_allow['es0'] and e_allow['es90']:
            xy_strains = np.abs(strains[:, 2])
            es_allow = min(abs(e_allow['es0']), abs(e_allow['es90']))
            margins[:, 2] = es_allow/xy_strains - 1
        # TODO: rotate strains and check 45 and -45 directions
        return margins

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
        brg = lek.LoadedHole(p, d, t, a_inv, theta=alpha)
        byp = lek.UnloadedHole(bypass, d, t, a_inv)
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

        """
        margins = []
        strains = self.strains(bearing, bypass, rc=rc, num=num)
        return self._strain_margins(strains)


