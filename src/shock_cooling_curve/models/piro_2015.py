# INTERNAL IMPORTS
from shock_cooling_curve.supernova import Supernova
import numpy as np
from shock_cooling_curve.utils import utils


class PIRO_2015(Supernova):
    """
    Class: Wraps around `supernova`, and inherits all `supernova` functionality.
    Produces synthetic photometry for shock-cooling emissions assuming the analytical
    shock-cooling model presented in Piro A.L (2015) [https://iopscience.iop.org/article/10.1088/2041-8205/808/2/L51].
    """

    def __init__(self, config_file, path_storage=None):
        """Initializes PIRO_2015 object.

        Args:
            config_file (str): name of configuration initialization file (.ini format)
            path_storage (str, optional): path to where the config file and photometry file (.csv)
            is stored. Defaults to None (current working directory).
        """
        
        self.initial = [11, 0.01, 0.1]
        self.lower_bounds = [0.1, 0.0001, 0.0]
        self.upper_bounds = [500, 1, 2]
        self.units = {'re': 'R_sun', 'me': 'M_sun', 'off': 'days'}
        self.scale = {'re': 1, 'me': 1, 'off': 1}
        self.display_name = "Piro (2015)"
        super().__init__(config_file, path_storage)

    def _tp(self, me, k):
        return 0.9 * ((k / 0.34) ** 0.5) * (self.bestek3 ** -0.25) * (self.mcore ** 0.17) * ((me / 0.01) ** 0.57)  # days

    def _ee(self, me):
        return 4e49 * self.bestek3 * (self.mcore ** -0.7) * ((me / 0.01) ** 0.7)

    def _vel(self, me):
        return 1e5 * 86400.0 * 2e9 * (self.bestek3 ** 0.5) * (self.mcore ** -0.35) * ((me / 0.01) ** -0.15)  # cm/s

    def luminosity(self, t, re, me, ve=None):
        """
        Analytical formalism for luminosity in Piro 2015.
        """
        self._t = t
        self._re = re
        self._me = me
        k = self.kappa

        te = (re * utils.rsun) / self._vel(me)
        L = ((te * self._ee(me)) /  self._tp(me, k) ** 2.) * \
            np.exp((-t * (t + 2. * te)) / (2. * self._tp(me, k) ** 2.))

        R = (re * utils.rsun) + (self._vel(me) * t / 86400.0)

        R = np.array([R])

        T = (L / (4. * np.pi * (R ** 2.) * utils.sigma)) ** 0.25

        T = np.array([T])
        # print(r, T)

        return R, T

    def f_lam(self, lam, r, t):
        return (np.pi / (self.dsn ** 2)) * (r ** 2) * (self.BB_lam(lam, t))  # ergs/ s / Ang.
