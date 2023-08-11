from shock_cooling_curve.supernova import *
from shock_cooling_curve.utils import utils

class SW_RSG(Supernova):
    def __init__(self, config_file, path_storage=None):
        super().__init__(config_file, path_storage)
        # sapir waxman: m = solar masses; r = r*10^13/rsun; v = 10^9cm/s
        self.units = {'re': 'R_sun', 'me': 'M_sun', 've': '1e9 cm/s', 'Off': 'days'}
        # TODO: ASK WYNN IF THIS SCALING IS RIGHT
        self.scale = {'re': 1e13/utils.rsun, 'me': 1, 've': 1*10**8.5/(10**9), 'Off': 1}

        self.display_name = 'Sapir & Waxman [n = 1.5]' # usually class.model_name returns str(class name)
        self.initial = [2, 0.5, 2, 0.01]
        self.lower_bounds = [0.01, 0.01, 0.01, 0.001]
        self.upper_bounds =[10, 10, 10, 0.5]

    def luminosity(self, t, Re, Me, ve):
        M = self.mcore + Me ## solar mass
        k = 0.2 / 0.34  # 1.
        fp = (Me / self.mcore) ** 0.5  # n=3/2
        vs = (ve * 1e9) / (10 ** 8.5) ## *10^4
        # R = re/1e13
        L = 1.88 * 1e42 * (((vs * (t ** 2.)) / (fp * M * k)) ** (-0.086)) * (vs ** 2. * Re / k) * np.exp(
            -(1.67 * t / (19.5 * (k * Me * vs ** -1) ** 0.5)) ** 0.8)
        T = 2.05 * 1e4 * (((vs ** 2. * t ** 2.) / (fp * M * k)) ** 0.027) * ((Re ** 0.25) / (k ** 0.25)) * (t ** -0.5)
        R = (L / (4. * np.pi * (T ** 4.) * utils.sigma)) ** 0.5 ##**10^13/R_sun
        return np.array([R]), np.array([T])