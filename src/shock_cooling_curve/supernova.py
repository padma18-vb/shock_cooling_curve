# REQUIRED IMPORTS

# PATH MANIPULATION
import os.path
import os
# HIDING LONG PROCESSING OR OUTPUT MESSAGES
from IPython.utils import io

# ARRAY OPERATIONS
import numpy as np

# IMPORT UTILS
# INTERNAL IMPORT
from shock_cooling_curve.utils import utils
# DATAFRAME HANDLING
import pandas as pd

# READING IN USER INPUT
import configparser

# ADDING EXTINCTION EFFECT TO PHOTOMETRY
import extinction

# TO CREATE SYNTHETIC PHOTOMETRY
import pysynphot as S

# IMPORT RE
import re as regex

import importlib.resources as pkg_resources


# STORING FILTER DIRECTORY


class Supernova(object):
    """
    Supernova class accepts information about target in dictionary format.
    Edit config.ini file under target's data directory to generate dictionary
    and create a supernova object.
    Reduces data, creates synthetic photometry for magnitude values based on
    specified mass and radius of envelope (and velocity of shock wave).
    """

    def __init__(self, config_file, path_to_storage=None):
        """
        Initializes supernova object.
        :param config_file: Name of config file.
        :param path_to_storage: Path to where the config file and data file are stored. Assumed to be local directory.
        :return: None
        """
        self.config_file = config_file
        self.path_to_storage = path_to_storage

        self.folder = regex.split("/", self.path_to_storage)[-1]
        d = self._make_dict()
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Supernova(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Supernova(b) if isinstance(b, dict) else b)
        self._controls()

    def _make_dict(self):
        """
        This function parses the config file into a usable dictionary.
        :return: dictionary containing key-value pairs read in from config file.
        """

        config_file = (os.path.join(self.path_to_storage, self.config_file))
        config = configparser.ConfigParser()
        config.read(config_file)
        details_dict = {}
        details_dict['folder'] = self.folder

        for key, val in config.items('DEFAULT'):
            if not regex.search('[a-zA-Z]', val):
                details_dict[key] = float(val)
            else:
                details_dict[key] = val
                # assert val[-3:] == 'csv', 'File must be .csv'
        return details_dict

    def _controls(self):
        """Helper function used to carry out initial data reduction operations.
        All main operations are called here to switch on and off functionality as needed.
        """
        self._assert_params()
        self._build_df()
        self._build_bandpass()

    def _assert_params(self):
        """Helper function used to calibrate and maintain global parameters. Configures input data.
        """
        self.a_v = self.ebv_mw * 3.1
        self.a_v_host = self.ebv_host * 3.1
        self.dsn = self.dist_sn * 3.086e+24  # cm
        self.model_name = self.__class__.__name__
        try:
            self.params = list(self.units.keys())
            self.n_params = len(self.params)
        except:
            print('None of the existing models (PIRO 2015/2020 or S&W 2017) have been used to create this object.')

        # set table input columns
        self.set_mag_colname()
        self.set_flt_colname()
        self.set_date_colname()
        self.set_magerr_colname()

        self.rmag_colname = 'RMAG'
        self.shift_date_colname = 'MJD_S'
        self.filter_info = utils.filter_info

    def make_path(self, filename):
        """Helper function used to store any generated data in the same directory as data source
        Args:
            filename (str): Name of generated file. Example: test_plot.png

        Returns:
            Path-like: OS Path to data origin folder with new file joined to the path.
        """
        return os.path.join(self.path_to_storage, filename)

    def set_mag_colname(self, name='MAG'):
        """Sets the magnitude column name. Defaults to "MAG", assumes that initial input data has "MAG" column,
        but once the data is read in, column names can be changed and reassigned using this function.

        Args:
            name (str, optional): Name of magnitude column. Defaults to 'MAG'.
        """
        self.mag_colname = name

    def set_magerr_colname(self, name='MAGERR'):
        """Sets the magnitude error column name. Defaults to "MAGERR", assumes that initial input data has
        "MAGERR" column, but once the data is read in, column names can be changed and reassigned using this function.

        Args:
            name (str, optional): Name of magnitude error column. Defaults to 'MAGERR'.
        """
        self.magerr_colname = name

    def set_date_colname(self, name='MJD'):
        """Sets the date (in MJD) column name. Defaults to "MJD", assumes that initial input data has
        "MJD" column, but once the data is read in, column names can be changed and reassigned using this function.

        Args:
            name (str, optional): Name of date column name. Defaults to 'MJD'.
        """
        self.date_colname = name

    def set_flt_colname(self, name='FLT'):
        """Sets the filter column name. Defaults to "FLT", assumes that initial input data has
        "FLT" column, but once the data is read in, column names can be changed and reassigned using this function.

        Args:
            name (str, optional): Name of filter column name. Defaults to 'FLT'.
        """
        self.flt_colname = name

    def set_vega_colname(self, name='Vega'):
        """Sets the boolean valued vega column name. Defaults to "Vega", assumes that initial input data has
        "Vega" column, but once the data is read in, column names can be changed and reassigned using this function.

        Args:
            name (str, optional): Name of vega flag column name. Defaults to 'Vega'.
        """
        self.vega_colname = name

    def convertAB_Vega(self, mag, flt, vega):
        """Helper function used to convert Vega magnitudes to AB.

        Args:
            mag (str): observed photometry value
            flt (str): filter in which measurement is made
            vega (int): boolean value: 1 = vega magnitude; 0 = AB magnitude.

        Returns:
            _type_: _description_
        """
        if vega:
            mag += self.filter_info.loc[flt]['AB - Vega']
            return mag
        return mag

    def reduce(self, mag, filt):
        """Helper function to apply reddening to observed photometry.

        Args:
            mag (str): observed photometry value
            filt (str): filter in which measurement is made

        Returns:
            int: reduced photometry value
        """
        wvl = np.array([utils.get_mapping('flt', filt, 'Effective Wavelength')])
        m_dr = mag - extinction.fitzpatrick99(wvl, self.a_v, r_v=3.1, unit='aa')
        m_dr = m_dr - extinction.fitzpatrick99(wvl, self.a_v_host, r_v=3.1, unit='aa')
        return m_dr[0]

    def add_red_mag(self, df):
        """
        Modifies original data frame.
        Converts Vega mags to AB mags.
        Applies Fitzpatrick 99 Extinction to AB mags assuming R_v = 3.1

        Args:
            df (pd.DataFrame): table containing photometry
        
        Returns:
            pd.DataFrame: modified dataframe with "RMAG" column containing extinction-correct AB mags.
        """
        ext_app = df.apply(
            lambda x: self.reduce(x[self.mag_colname], x[self.flt_colname]), axis=1)
        df[self.rmag_colname] = ext_app
        vega_conv = df.apply(
            lambda x: self.convertAB_Vega(x[self.rmag_colname], x[self.flt_colname], x['Vega']), axis=1)
        df[self.rmag_colname] = vega_conv
        return df

    def _build_df(self):
        """Helper function to filter only the data during shock cooling.
        """
        # read in data from photometry file
        self.data_all = pd.read_csv(self.make_path(self.filename))
        # convert vega to AB mag and apply extinction
        reduced_df = self.add_red_mag(self.data_all)
        # add column "MJD_S" which converts MJD dates to SN start reference frame (starting from 0)
        reduced_df = reduced_df.assign(MJD_S=reduced_df[self.date_colname] - self.start_sn)
        self.data_all = reduced_df.copy(deep=True)
        self.data_all = self.data_all[self.data_all['MJD_S'] > 0]
        reduced_df = reduced_df[reduced_df[self.date_colname] <= self.end_sc]
        reduced_df = reduced_df[reduced_df[self.date_colname] >= self.start_sn]

        sorter = list(self.filter_info.index)
        sorted_flt = dict(zip(sorter, range(len(sorter))))
        reduced_df['ind'] = reduced_df['FLT'].map(sorted_flt)
        reduced_df = reduced_df.sort_values('ind')
        self.reduced_df = reduced_df.drop('ind', axis=1)
        self.reduced_df = self.reduced_df[self.reduced_df['MAGERR'] > 0]

        self.reduced_df.to_csv(self.make_path('reduced_data.csv'))

    @property
    def extinction_applied_full_data(self):
        """
        Returns:
            pd.DataFrame: All photometry data supplied with extinction corrected 
            photometry in a separate "RMAG" column
        """
        return self.data_all

    @property
    def extinction_applied_data_during_shock_cooling(self):
        """
        Returns:
            pd.DataFrame: All photometry data during shock cooling with extinction corrected 
            photometry in a separate "RMAG" column. This data is used in all fitting procedures.
        """
        return self.reduced_df

    def get_filts(self):
        """Queries input dataframe for all filters used to observe the target.

        Returns:
            pd.Series: All unique filters for supplied target.
        """
        return pd.unique(self.reduced_df[self.flt_colname])

    def _build_bandpass(self):
        """Helper function to associate filters with pysynphot passbands.
        """
        filter_files = {}  # key: flt, value: array containing filter values
        self.bandpass_dict = {}  # key: flt, value: bandpass
        filts = self.get_filts()
        for flt in filts:
            bandpass = self.filter_info.loc[flt]['Bandpass']
            try:

                flt_filename = f'{bandpass}.{flt}.dat'
                filter_path = pkg_resources.path(f'shock_cooling_curve.filters', f'{flt_filename}')
                with filter_path as p:
                    filter_files[flt] = np.genfromtxt(p)

                arr = filter_files[flt]
                self.bandpass_dict[flt] = S.ArrayBandpass(arr[:, 0], arr[:, 1], name=flt)
            except (FileNotFoundError, IOError):
                self.bandpass_dict[flt] = S.ObsBandpass(f'{bandpass},{flt}')

    # only called when fitting occurs
    def _mags_per_filter(self, times: [float], filtName=None, **kwargs):
        """_summary_
        :param re: Radius of envelope in R_sun
        :param me: Mass of envelope
        :param ve: velocity of propagation (how do we say this) how fast the wave is moving out
        Args:
            times ([float]): observation times
            filtName (str, optional): filter name. Defaults to None.
            **kwargs

        Returns:
            array-like: all mags for specified times fitted assuming a BB model according to specified filter
        """

        R, T = self.luminosity(times, **kwargs)

        R = R.flatten()
        T = T.flatten()
        with io.capture_output():
            mags = np.array([])
            for i in range(len(T)):
                bb = S.BlackBody(T[i])
                bb.convert('flam')  # erg / s / cm^2 / AA
                l = np.linspace(np.min(bb.wave), np.max(bb.wave), 1000)
                norm = (bb.integrate(fluxunits='flam') ** -1.) * np.trapz(utils.BB_lam(l, T[i]), l)
                bb_norm = bb * norm
                Flam = (np.pi / (self.dsn ** 2)) * (R[i] ** 2) * bb_norm  # ergs/ s / Ang.

                bandpass = self.bandpass_dict[filtName]
                obs = S.Observation(Flam, bandpass)
                mag = obs.effstim('abmag')
                mags = np.append(mags, mag)
                # return estimate of distance
        return mags

    def get_all_mags(self, times, *args):
        """Obtains synthetic photometry values given times and fitting parameters (e.g.: re, me, ve, off).

        Args:
            times (array-like): observed time values

        Returns:
            array-like: synthetic photometry values
        """

        filters = np.array(self.reduced_df['FLT'])
        if len([*args]) == 3:
            re, me, off = args
            ve = None
        else:
            re, me, ve, off = args

        # with io.capture_output() as captured:
        # mjd date, of - free parameter
        all_mags = np.array([])
        for i in range(len(times)):
            flt = filters[i]
            time = times[i] - off
            val = self._mags_per_filter(time, filtName=flt, re=re, me=me,
                                        ve=ve)

            all_mags = np.append(all_mags, val)
        return all_mags

    def write_to_file(self):
        return

    def save_bounds(self, config_file):
        """Saves input bounds for fitting to configuration file

        Args:
            config_file (.ini file): supernova config file
        """
        config_file = self.make_path(config_file)
        config = configparser.ConfigParser()
        config.read(config_file)
        section = f'{self.model_name} BOUNDS'
        config.add_section(section)
        try:
            config.set(section, 'initial', str(self.initial))
            config.set(section, 'lower_bounds', str(self.lower_bounds))
            config.set(section, 'upper_bounds', str(self.upper_bounds))

        except:
            print('You have not entered bounds for this object yet.')
            return

        with open(config_file, 'w') as configfile:
            config.write(configfile)

    def get_curvefit_values(self):
        """Returns best-fit curve fit values

        Returns:
            tuple: contains two dictionaries - [keys]: param names and [values]: best fit values
        """
        try:
            return self.CF_fitted_params, self.CF_fitted_errors
        except:
            print("This object has not been modelled using `scipy.opt.curve_fit`. Create a Fitter object and model "
                  "using Fitter.simple_curve_fit().")
        return

    def get_MCMC_values(self):
        """Returns best-fit MCMC values

        Returns:
            tuple: contains two dictionaries - [keys]: param names and [values]: best fit values
        """
        try:
            return self.MCMC_fitted_params, self.MCMC_fitted_errors
        except:
            print("This object has not been modelled using `emcee`. Create a Fitter object and model using "
                  "Fitter.MCMC_fitter().")

    def get_model_performance(self, params):
        """Computes reduced chi-squared values given fit-parameters

        Args:
            params (dict): parameter name keys mapped to fit values

        Returns:
            float: reduced chi-squared value of fit
        """
        xdata_phase = np.array(self.reduced_df[self.shift_date_colname])
        ydata_mag = np.array(self.reduced_df[self.rmag_colname])
        yerr_mag = np.array(self.reduced_df[self.magerr_colname])

        try:
            yfit_mag = self.get_all_mags(xdata_phase, params['re'], params['me'], params['ve'], params['off'])
        except KeyError:
            yfit_mag = self.get_all_mags(xdata_phase, params['re'], params['me'], params['off'])

        dof = len(xdata_phase) - len(params)
        resid = (ydata_mag - yfit_mag) / yerr_mag
        chisq = np.dot(resid, resid)
        return chisq / dof
