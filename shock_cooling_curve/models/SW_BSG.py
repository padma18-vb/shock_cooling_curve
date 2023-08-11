from shock_cooling_curve.supernova import *
from shock_cooling_curve.utils import utils

class SW_BSG(Supernova):
    def __init__(self, config_file, path_storage=None):
        super().__init__(config_file, path_storage)
        self.set_model()
        self.units = {'re': 'R_sun', 'me': 'M_sun', 've': '1e9 cm/s', 'Off': 'days'}
        self.scale = {'re': 1e13/utils.rsun, 'me': 1, 've': 1*10**8.5/(10**9), 'Off': 1}

        self.display_name = 'Sapir & Waxman [n = 3]' # usually class.model_name returns str(class name)

        self.initial =[2.,0.5, 2., 0.04]
        self.lower_bounds = [0.01, 0.001, 0.01, 0.01]
        self.upper_bounds = [10, 3. , 10., 0.2]

    def luminosity(self, t, Re, Me, ve):
        M = self.mcore + Me
        k = 0.2 / 0.34  # 1.
        fp = (Me / self.mcore) * 0.08  # n=3
        vs = (ve * 1e9) / (10 ** 8.5)
        # R = re/1e13
        L = 1.66 * 1e42 * (((vs * (t ** 2.)) / (fp * M * k)) ** (-0.175)) * (vs ** 2. * Re / k) * np.exp(
            -((4.57 * t) / (19.5 * (k * Me * (vs ** -1.)) ** 0.5)) ** (0.73))
        T = 1.96 * 1e4 * (((vs ** 2. * t ** 2.) / (fp * Me * k)) ** 0.016) * ((Re ** 0.25) / (k ** 0.25)) * (t ** -0.5)
        R = (L / (4. * np.pi * (T ** 4.) * utils.sigma)) ** 0.5
        return np.array([R]), np.array([T])