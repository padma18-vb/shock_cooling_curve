# INTERNAL IMPORTS
from src.shock_cooling_curve.supernova import *

class PIRO_2015(Supernova):

    """
    Class: Wraps around `supernova`, and inherits all `supernova` functionality.
    Produces synthetic photometry for shock-cooling emissions assuming the analytical
    shock-cooling model presented in Piro A.L (2015) [https://iopscience.iop.org/article/10.1088/2041-8205/808/2/L51].
    """

    def __init__(self, config_file, path_storage=None):
        
        self.display_name = "Piro (2015)"
        self.initial = [11, 0.01, 0.1]
        self.lower_bounds = [0.1, 0.0001, 0.0]
        self.upper_bounds = [500, 1, 2]
        self.units = {'Re': 'R_sun', 'Me': 'M_sun', 'Off': 'days'}
        self.scale = {'Re': 1, 'Me': 1, 'Off': 1}
        super().__init__(config_file, path_storage)



    def luminosity(self, t, Re, Me, ve=None, k=0.2):

        """
        Luminosity 
        """

        def tp(Esn, Mc, Me, k):
            return 0.9 * ((k / 0.34) ** 0.5) * ((Esn) ** -0.25) * ((Mc) ** 0.17) * ((Me / (0.01)) ** 0.57)  # days
        # is me in units of 0.01?

        def Ee(Esn, Mc, Me):
            return 4e49 * (Esn) * ((Mc) ** -0.7) * ((Me / (0.01)) ** 0.7)

        def ve(Esn, Mc, Me):
            return 1e5 * (86400.0) * (2e9) * ((Esn) ** 0.5) * ((Mc) ** -0.35) * ((Me / (0.01)) ** -0.15)  # cm/s

        te = (Re * utils.rsun) / ve(self.bestek3, self.mcore, Me)
        L = ((te * Ee(self.bestek3, self.mcore, Me)) / tp(self.bestek3, self.mcore, Me, k) ** 2.) * \
            np.exp((-t * (t + 2. * te)) / (2. * tp(self.bestek3, self.mcore, Me, k) ** 2.))

        R = (Re * utils.rsun) + (ve(self.bestek3, self.mcore, Me) * t / (86400.0))

        R = np.array([R])

        T = (L / (4. * np.pi * (R ** 2.) * utils.sigma)) ** 0.25

        T = np.array([T])
        # print(R, T)

        return R, T

    def f_lam(self, lam, R, T):
        return (np.pi / (self.dsn ** 2)) * (R ** 2) * (self.BB_lam(lam, T))  # ergs/ s / Ang.
