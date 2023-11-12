from shock_cooling_curve.supernova import *
from shock_cooling_curve.utils import utils


class PIRO_2020(Supernova):
    """ 
    Class: Wraps around `supernova`, and inherits all `supernova` functionality.
    Produces synthetic photometry for shock-cooling emissions assuming the analytical
    shock-cooling model presented in Piro A.L (2020) [https://doi.org/10.3847/1538-4357/abe2b1].
    """
    # these number were derived using numerical simulations
    n = 10  # density index; typical value = 10
    K = 0.119  # typical value; K = [(n - 3) * (3 - delta)]/[4*pi * (n - delta)]
    delta = 1.1  # typical value

    def __init__(self, config_file, path_storage=None):
        """Initializes PIRO_2020 object.

        Args:
            config_file (str): name of configuration initialization file (.ini format)
            path_storage (str, optional): path to where the config file and photometry file (.csv)
            is stored. Defaults to None (current working directory).
        """
        self.units = {'re': 'R_sun', 'me': 'M_sun', 've': '1e9 cm/s', 'off': 'days'}
        self.scale = {'re': 1, 'me': 1, 've': 1, 'off': 1}
        self.display_name = 'Piro (2020)'
        self.initial = [30, 0.5, 2.1, 0.1]
        self.lower_bounds = [0.1, 0.01, 0.1, 0.0]
        self.upper_bounds = [1000, 1, 10, 0.5]
        super().__init__(config_file, path_storage)

    def _td(self, m, v):
        """Time at which diffusion depth reaches vt.

        Args:
            m (float): mass in g
            v (float): velocity in cm/s

        Returns:
            float: time in s
        """
        numerator = 3 * self.kappa * self.K * m
        denominator = (self.n - 1) * v * utils.c
        return (numerator / denominator) ** 0.5  # seconds

    def _tph(self, m, v):
        """Time at which photosphere depth reaches vt

        Args:
            m (float): mass in g
            v (float): velocity in cm/s

        Returns:
            float: time in s
        """
        numerator = 3 * self.kappa * self.K * m
        denominator = 2 * (self.n - 1) * v ** 2
        return (numerator / denominator) ** 0.5  # seconds

    def _rph_early(self, t, m, v):
        """Radius before _tph

        Args:
            t (float): time in seconds
            m (float): mass in g
            v (float): velocity in cm/s

        Returns:
            float: radius in cm
        """
        time_term = self._tph(m, v) / t
        power = 2 / (self.n - 1)

        return (time_term ** power) * v * t

    def _rph_late(self, t, m, v):
        """Radius after _tph

        Args:
            t (float): time in seconds
            m (float): mass in g
            v (float): velocity in cm/s

        Returns:
            float: radius in cm
        """
        power = -1 / (self.delta - 1)
        first_term = (self.delta - 1) / (self.n - 1)
        time_term = (t / self._tph(m, v)) ** 2
        return (((first_term * (time_term - 1)) + 1) ** power) * v * t

    def luminosity(self, t, re, me, ve):
        """Computes Radius and Temperature under the Piro 2020 model. Luminosity is computed
        using radius and temperature assuming a blackbody relationship. Typical values in this
        section are assumed from Chevalier & Soker (1989) 
        [https://ui.adsabs.harvard.edu/abs/1989ApJ...341..867C/abstract]

        Args:
            t (float): time in MJD (starting from SN start time)
            re (float): envelope radius
            me (float): envelope mass
            ve (float): shock velocity

        Returns:
            tuple: radius and temperature of shock cooling emission at provided time
        """

        vt_cgs = ve * 1e9  # cm / s
        me_cgs = me * utils.msun  # solar mass to g
        re_cgs = re * utils.rsun  # solar radii to cm

        t_cgs = np.array([t]).flatten() * 86400  # days to seconds
        l_cgs = np.ones(len(t_cgs))

        prefactor = (np.pi / 3) * ((self.n - 1) / (self.n - 5)) * ((utils.c * re_cgs * vt_cgs ** 2) / self.kappa)
        t_smaller_td = np.where(t_cgs < self._td(me_cgs, vt_cgs))
        t_larger_td = np.where(t_cgs >= self._td(me_cgs, vt_cgs))
        l_cgs[t_smaller_td] = np.array(prefactor * (self._td(me_cgs, vt_cgs) / t_cgs) ** (4 / (self.n - 2)))[
            t_smaller_td]
        l_cgs[t_larger_td] = np.array(prefactor * np.exp(
            -0.5 * (((t_cgs / self._td(me_cgs, vt_cgs)) ** 2) - 1)))[t_larger_td]

        r_cgs = np.ones(len(t_cgs))

        t_smaller_tph = np.where(t_cgs < self._tph(me_cgs, vt_cgs))
        t_larger_tph = np.where(t_cgs >= self._tph(me_cgs, vt_cgs))
        r_cgs[t_smaller_tph] = np.array(self._rph_early(t_cgs, me_cgs, vt_cgs))[t_smaller_tph]
        r_cgs[t_larger_tph] = np.array(self._rph_late(t_cgs, me_cgs, vt_cgs))[t_larger_tph]

        temp_cgs = (l_cgs / (4 * np.pi * r_cgs ** 2 * utils.sigma)) ** (1 / 4)

        return r_cgs, temp_cgs
