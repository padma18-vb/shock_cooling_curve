from shock_cooling_curve.supernova import *
from shock_cooling_curve.utils import utils


class SW_BSG(Supernova):
    """Wraps around `supernova`, and inherits all `supernova` functionality.
    Produces synthetic photometry for shock-cooling emissions assuming the analytical
    shock-cooling model presented in Sapir & Waxman (2017) [https://iopscience.iop.org/article/10.3847/1538-4357/aa64df].
    """

    def __init__(self, config_file, path_storage=None):
        """Initializes SW_BSG object.

        Args:
            config_file (str): name of configuration initialization file (.ini format)
            path_storage (str, optional): path to where the config file and photometry file (.csv)
            is stored. Defaults to None (current working directory).
        """
        self.units = {'re': 'R_sun', 'me': 'M_sun', 've': '1e9 cm/s', 'off': 'days'}
        self.scale = {'re': 1e13 / utils.rsun, 'me': 1, 've': 1, 'off': 1}

        self.display_name = 'Sapir & Waxman (2017) [n = 3]'  # usually class.model_name returns str(class name)

        self.initial = [2., 0.5, 2., 0.04]
        self.lower_bounds = [0.01, 0.001, 0.01, 0.01]
        self.upper_bounds = [10, 3., 10., 0.2]

        super().__init__(config_file, path_storage)

    def luminosity(self, t, re, me, ve):
        """Applies the analytic model described in Sapir & Waxman (2017) for a blue supergiant 
        (i.e density slope = 3).

        Args:
            t (array-like): time of observation (from the supernova start time)
            re (float): radius of progenitor envelope
            me (float): mass of progenitor envelope
            ve (float): velocity of shock

        Returns:
            : tuple of two arrays containing radius and temperature values
        """

        M = self.mcore + me  # solar mass
        k = self.kappa / 0.34  # 1.
        fp = 0.08 * (me / self.mcore)
        vs = (ve * 1e9) / (10 ** 8.5)  # *10^4
        # lum terms

        lterm1 = ((vs * t ** 2) / (fp * M * k)) ** (-0.175)
        lterm2 = ((vs ** 2) * re) / k

        num = 4.57 * t
        den = 19.5 * (k * me * (1 / vs)) ** (1 / 2)
        lterm3 = np.exp(- ((num / den) ** 0.73))
        L = 1.66 * 1e42 * lterm1 * lterm2 * lterm3

        tterm1 = (((vs * t) ** 2) / (fp * M * k)) ** 0.016
        tterm2 = ((re / k) ** 0.25) * t ** (-0.5)

        T = 1.96 * 1e4 * tterm1 * tterm2

        R = (L / (4. * np.pi * (T ** 4.) * utils.sigma)) ** 0.5  # **10^13/R_sun

        return np.array([R]), np.array([T])
