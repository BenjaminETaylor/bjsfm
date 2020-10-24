import numpy as np
import bjsfm.lekhnitskii as lek


class MaxStrain:
    """ A class for analyzing joint failure with using max strain failure theory.

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
        self.a_inv = np.linalg.inv(a)
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

    def _strain_margins(self, strain):
        """Calculates margins of safety"""
        raise NotImplementedError("Oops! Haven't implemented this yet.")

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
            strain = a_inv.dot(stress)
            margins.append(self._strain_margins(strain))
        return margins


