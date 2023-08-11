from shock_cooling_curve.supernova import *
from shock_cooling_curve.utils import utils


class PIRO_2020(Supernova):
    ''' Class: Wraps around `supernova`, and inherits all `supernova` functionality.
    Produces synthetic photometry for shock-cooling emissions assuming the analytical
    shock-cooling model presented in Piro A.L (2020) [https://doi.org/10.3847/1538-4357/abe2b1].
    '''

    n = 10  # density index; typical value = 10
    kap = 0.34  # cm^2/g; optical depth = 0.34 (assumes electron scattering opacity)
    K = 0.119  # typical value; K = [(n - 3) * (3 - delta)]/[4*pi * (n - delta)]
    delta = 1.1  # typical value

    def __init__(self, config_file, path_storage=None):
        super().__init__(config_file, path_storage)
        self.units = {'re': 'R_sun', 'me': 'M_sun', 've': '1e9 cm/s', 'Off': 'days'}
        self.scale = {'re': 1, 'me': 1, 've': 1, 'Off': 1}
        self.display_name = 'Piro (2020)'
        self.initial = [30, 0.5, 2.1, 0.1]
        self.lower_bounds = [0.1, 0.01, 0.1, 0.0]
        self.upper_bounds = [1000, 1, 10, 0.5]

    def _td(self, M, v):
        numerator = 3 * self.kap * self.K * M
        denominator = (self.n - 1) * v * utils.c
        return (numerator / denominator) ** 0.5  # seconds

    def _tph(self, M, v):
        numerator = 3 * self.kap * self.K * M
        denominator = 2 * (self.n - 1) * v ** 2
        return (numerator / denominator) ** 0.5  # seconds

    def _rph_early(self, t, M, v):
        time_term = self._tph(M, v) / t
        power = 2 / (self.n - 1)

        return (time_term ** power) * v * t

    def _rph_late(self, t, M, v):
        power = -1 / (self.delta - 1)
        first_term = (self.delta - 1) / (self.n - 1)
        time_term = (t / self._tph(M, v)) ** 2
        return (((first_term * (time_term - 1)) + 1) ** power) * v * t

    def luminosity(self, t, Re, Me, ve):
        '''Computes Radius and Temperature under the Piro 2020 model. Luminosity is computed
        using radius and temperature assuming a blackbody relationship. Typical values in this
        section are assumed from Chevalier & Soker (1989) [https://ui.adsabs.harvard.edu/abs/1989ApJ...341..867C/abstract]'''

        vt_cgs = ve * 1e9  # cm / s
        Me_cgs = Me * utils.msun  # solar mass to g
        Re_cgs = Re * utils.rsun  # solar radii to cm

        t_cgs = np.array([t]).flatten() * 86400  # days to seconds
        L_cgs = np.ones(len(t_cgs))

        prefactor = (np.pi / 3) * ((self.n - 1) / (self.n - 5)) * ((utils.c * Re_cgs * vt_cgs ** 2) / self.kap)

        L_cgs[t < self._td(Me_cgs, vt_cgs)] = prefactor * (self._td(Me_cgs, vt_cgs) / t_cgs) ** (4 / (self.n - 2))
        L_cgs[t >= self._td(Me_cgs, vt_cgs)] = prefactor * np.exp(
            -0.5 * (((t_cgs / self._td(Me_cgs, vt_cgs)) ** 2) - 1))

        R_cgs = np.ones(len(t_cgs))

        R_cgs[t < self._tph(Me_cgs, vt_cgs)] = self._rph_early(t_cgs, Me_cgs, vt_cgs)
        R_cgs[t >= self._tph(Me_cgs, vt_cgs)] = self._rph_late(t_cgs, Me_cgs, vt_cgs)

        T_cgs = (L_cgs / (4 * np.pi * R_cgs ** 2 * utils.sigma)) ** (1 / 4)

        return R_cgs, T_cgs

