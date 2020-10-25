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
    tension : array_like, optional
        1D 1x4 array [et0, et45, et90, et-45] tension strain allowables in all four directions
    compression : array_like, optional
        1D 1x4 array [ec0, ec45, ec90, ec-45] compression strain allowables in all four directions
    shear : array_like, optional
        1D 1x4 array [es0, es45, es90, es-45] shear strain allowables in all four directions

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
    tens : ndarray
        1D 1x4 array [et0, et45, et90, et-45] tension strain allowables in all four directions
    comp : ndarray
        1D 1x4 array [ec0, ec45, ec90, ec-45] compression strain allowables in all four directions
    shear : ndarray
        1D 1x4 array [es0, es45, es90, es-45] shear strain allowables in all four directions

    """

    def __int__(self, thickness, diameter,  a_matrix, tension=None, compression=None, shear=None):
        self.t = thickness
        self.r = diameter/2.
        self.a = np.array(a_matrix)
        self.a_inv = np.linalg.inv(self.a)
        self.tens = np.array(tension) if tension else None
        self.comp = np.array(compression) if compression else None
        self.shear = np.array(shear) if shear else None

    def _radial_points(self, step, num_pnts):
        """Calculates x, y points"""
        r = np.array([self.r + step] * num_pnts)
        theta = np.linspace(0, 2*np.pi, num=num_pnts, endpoint=False)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    @staticmethod
    def _strain_margins(strains, e1t=None, e1c=None, e2t=None, e2c=None, e12=None):
        """Calculates margins of safety"""
        margins = [np.nan]*3
        if e1t and e1c:
            # 0 deg
            margins[0] = e1t/strains[0] - 1 if strains[0] > 0. else e1c/strains[0] - 1
        if e2t and e2c:
            # 90 deg
            margins[1] = e2t/strains[1] - 1 if strains[1] > 0. else e2c/strains[1] - 1
        if e12:
            # shear
            margins[2] = e12/strains[2] - 1
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
        alpha = np.tan(bearing[1]/bearing[0])
        p = np.sqrt(np.sum(np.square(bearing)))
        x, y = self._radial_points(rc, num)
        brg = lek.LoadedHole(p, d, t, a_inv, theta=alpha)
        byp = lek.UnloadedHole(bypass, d, t, a_inv)
        byp_stress = byp.stress(x, y)
        brg_stress = brg.stress(x, y)
        return byp_stress + brg_stress

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
            2D numx3 array of margins of safety

        """
        margins = []
        a_inv = self.a_inv
        stresses = self.stresses(bearing, bypass, rc=rc, num=num)
        for stress in stresses:
            # x-direction
            strain = a_inv.dot(stress/self.t)
            e1t, e1c, e2t, e2c, e12 = self.tens[0], self.comp[0], self.tens[2], self.comp[2], self.shear[0]
            margins.append(self._strain_margins(strain, e1t=e1t, e1c=e1c, e2t=e2t, e2c=e2c, e12=e12))
        return margins


