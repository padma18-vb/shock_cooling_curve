#### REQUIRED IMPORTS ####

# PATH MANIPULATION
import os.path

# ARRAY OPERATIONS
import numpy as np

# IMPORT UTILS
# INTERNAL IMPORT
from shock_cooling_curve.utils import utils

# DATAFRAME HANDLING
import pandas as pd

# HIDING LONG PROCESSING OR OUTPUT MESSAGES
from IPython.utils import io

# READING IN USER INPUT
import configparser

# ADDING EXTINCTION EFFECT TO PHOTOMETRY
import extinction

# TO CREATE SYNTHETIC PHOTOMETRY
import pysynphot as S

# IMPORT RE
import re

from importlib import resources

# STORING FILTER DIRECTORY
filter_dir = 'shock_cooling_curve/filters/'



class Supernova(object):
    """
    Supernova class accepts information about target in dictionary format.
    Edit config.ini file under target's data directory to generate dictionary
    and create a supernova object.
    Reduces data, creates synthetic photometry for magnitude values based on
    specified mass and radius of envelope (and velocity of shock wave).
    """

    def __init__(self, config_file, path_to_storage = None):
        """
        Initializes supernova object.
        :param config_file: Name of config file.
        :param path_to_storage: Path to where the config file and data file are stored. Assumed to be local directory.
        :return: None
        """
        self.config_file = config_file
        self.path_to_storage = path_to_storage
        self.folder = re.split("/", self.path_to_storage)[-1]
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
            if not re.search('[a-zA-Z]', val):
                details_dict[key] = float(val)
            else:
                details_dict[key] = val
                # assert val[-3:] == 'csv', 'File must be .csv'
        return details_dict

    def _controls(self):
        """
        Helper function used to carry out initial data reduction operations.
        All main operations are called here to switch on and off functionality as needed.
        :return: None
        """
        self._assert_params()
        self._build_df()
        self._build_bandpass()

    def _assert_params(self):
        """
        Helper function used to calibrate and maintain global parameters. Configures input data.
        :return: None
        """
        self.a_v = self.ebv_mw * 3.1
        self.a_v_host = self.ebv_host * 3.1
        self.dsn = self.dist_sn * 3.086e+24  # cm
        self.model_name = self.__class__.__name__

        # set table input columns
        self.set_mag_colname()
        self.set_flt_colname()
        self.set_date_colname()
        self.set_magerr_colname()

        self.rmag_colname = 'RMAG'
        self.shift_date_colname = 'MJD_S'
        self.filter_info = utils.filter_info

    def make_path(self, filename):
        """
        Helper function used to store any generated data in the same directory as data source
        :param filename: Name of generated file. Example: test_plot.png
        :return: OS Path to data origin folder with new file joined to the path.
        """
        return os.path.join(self.path_to_storage, filename)

    def set_mag_colname(self, name='MAG'):
        """
        Setter: sets the magnitude column name. Defaults to "MAG", assumes that initial input data has "MAG" column,
        but once the data is read in, column names can be changed and reassigned using this function.
        """
        self.mag_colname = name

    def set_magerr_colname(self, name='MAGERR'):
        """
        Setter: sets the magnitude error column name. Defaults to "MAGERR", assumes that initial input data has
        "MAGERR" column, but once the data is read in, column names can be changed and reassigned using this function.
        """
        self.magerr_colname = name

    def set_date_colname(self, name='MJD'):
        """
        Setter: sets the date (in MJD) column name. Defaults to "MJD", assumes that initial input data has
        "MJD" column, but once the data is read in, column names can be changed and reassigned using this function.
        """
        self.date_colname = name

    def set_flt_colname(self, name='FLT'):
        """
        Setter: sets the filter column name. Defaults to "FLT", assumes that initial input data has
        "FLT" column, but once the data is read in, column names can be changed and reassigned using this function.
        """
        self.flt_colname = name

    def set_vega_colname(self, name='Vega'):
        """
        Setter: sets the boolean valued vega column name. Defaults to "Vega", assumes that initial input data has
        "Vega" column, but once the data is read in, column names can be changed and reassigned using this function.
        """
        self.vega_colname = name

    def convertAB_Vega(self, flt, mag, vega):
        """
        :param flt: Filter in which measurement is made
        :param mag: Magnitude measurement
        :param vega: Boolean value: 1 = vega magnitude; 0 = AB magnitude.
        Helper function used to convert Vega magnitudes to AB.
        """
        if vega:
            mag += self.filter_info.loc[flt]['AB - Vega']
            return mag
        return mag

    def add_red_mag(self, df):
        """

        """
        # makes no change to original data frame for future reference for user
        dfnew = df.copy()
        mags = np.array(dfnew[self.mag_colname])
        dfnew[self.mag_colname] = dfnew.apply(
            lambda x: self.convertAB_Vega(x[self.flt_colname], x[self.mag_colname], x['Vega']), axis=1)
        filts = np.array(dfnew[self.flt_colname])
        mag_to_filt = dict(zip(mags, filts))

        def reduce(mag):
            filt = mag_to_filt[mag]
            wvl = np.array([utils.get_mapping('flt', filt, 'Effective Wavelength')])
            m_dr = mag - extinction.fitzpatrick99(wvl, self.a_v, r_v=3.1, unit='aa')
            m_dr = m_dr - extinction.fitzpatrick99(wvl, self.a_v_host, r_v=3.1, unit='aa')
            return m_dr

        reduced_mags = np.array([reduce(mag) for mag in mags])
        dfnew[self.rmag_colname] = reduced_mags

        return dfnew

    def _build_df(self):
        self.data_all = pd.read_csv(self.make_path(self.filename))

        reduced_df = self.add_red_mag(self.data_all)
        reduced_df = reduced_df.assign(MJD_S=reduced_df[self.date_colname] - self.start_sn)
        self.data_all = reduced_df.copy()
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
    def get_extinction_applied_full_data(self):
        return self.data_all

    @property
    def get_data_during_shock_cooling(self):
        return self.reduced_df

    def get_filts(self):
        '''
        Queries input dataframe for all filters used to observe the target.
        :return: pandas Series object with all unique filters for supplied target.
        '''
        return pd.unique(self.reduced_df[self.flt_colname])

    def _build_bandpass(self):
        filter_files = {}  # key: flt, value: array containing filter values
        self.bandpass_dict = {}  # key: flt, value: bandpass
        filts = self.get_filts()
        for flt in filts:
            bandpass = self.filter_info.loc[flt]['Bandpass']
            try:
                flt_filename =f'{filter_dir}{bandpass}_Filters/{bandpass}.{flt}.dat'
                filter_files[flt] = np.genfromtxt(flt_filename)

                arr = filter_files[flt]
                self.bandpass_dict[flt] = S.ArrayBandpass(arr[:, 0], arr[:, 1], name=flt)
            except (FileNotFoundError, IOError):
                self.bandpass_dict[flt] = S.ObsBandpass(f'{bandpass},{flt}')

    # only called when fitting occurs
    def _mags_per_filter(self, times: [float], filtName = None, **kwargs):
        '''
        :param times: observation times
        :param filtName: filter name
        :param re: Radius of envelope in R_sun
        :param me: Mass of envelope
        :param ve: velocity of propagation (how do we say this) how fast the wave is moving out

        :return: array of all mags for specified times fitted assuming a BB model according to specified filter
        '''

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
        '''
        Obtains 
        :param times: Times at which data is available
        :param Re: Radius of envelope
        :param Me: Mass of envelope
        :param ve: Velocity of wave moving out
        :param of: Offset for observation time
        :return: array of all mags at specified time according to filter
        '''
        filters = np.array(self.reduced_df['FLT'])
        if len([*args]) == 3:
            re, me, of = args
            ve = None
        else:
            re, me, ve, of = args

        # with io.capture_output() as captured:
        # mjd date, of - free parameter
        all_mags = np.array([])
        for i in range(len(times)):
            flt = filters[i]
            time = times[i] - of
            val = self._mags_per_filter(time, filtName=flt, Re=re, Me=me,
                                       ve=ve)

            all_mags = np.append(all_mags, val)
        return all_mags


    def write_to_file(self):
        return

    def save_bounds(self, config_file):
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
        try:
            return self.NL_fitted_params, self.NL_fitted_errors
        except:
            print('No fitted values for curve fit available. Call Fitter(this_object, "Non-linear Curve Fit")'
                  'to obtain these values.')
        return

    def get_MCMC_values(self):
        return self.MCMC_fitted_params, self.MCMC_fitted_errors



