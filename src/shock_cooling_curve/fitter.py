import numpy as np
import scipy.optimize as opt
import emcee as em
import pandas as pd
import os
from multiprocessing import Pool


class Fitter:
    """
    Used for fitting a `Supernova` object. Non-linear least squares fitting and MCMC sampling are currently implemented.
    """

    def __init__(self, sn_obj):

        self.sn_obj = sn_obj
        self.xdata_phase = np.array(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname])
        self.ydata_mag = np.array(self.sn_obj.reduced_df[self.sn_obj.mag_colname])
        self.yerr_mag = np.array(self.sn_obj.reduced_df[self.sn_obj.magerr_colname])

        self.func = self.sn_obj.get_all_mags
        self.params = self.sn_obj.params
        self.n_params = self.sn_obj.n_params

    def simple_curve_fit(self):

        popt, pcov = opt.curve_fit(
            f=self.func,
            xdata=self.xdata_phase,
            ydata=self.ydata_mag,
            p0=self.sn_obj.initial,
            sigma=self.yerr_mag,
            bounds=(self.sn_obj.lower_bounds, self.sn_obj.upper_bounds)
        )
        # we're making some assumptions here - make sure you note down what those are
        sigma = np.sqrt(np.diag(pcov))

        self.sn_obj.CF_fitted_params = dict(zip(self.params, popt))
        self.sn_obj.CF_fitted_errors = dict(zip(self.params, sigma))


        return popt, sigma
    
    def _get_val(self, param, value_dict, err_dict):
        print(f'Units: {self.sn_obj.units[param]}')
        return value_dict[param] * self.sn_obj.scale[param], err_dict[param] * self.sn_obj.scale[param]
        

    @property
    def get_cf_RE(self):
        """
        :return: Best fit non-linear least squares curve fit value for envelope mass in solar mass.
        """
        param = "Re"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors
        
        return self._get_val(param, vdict, edict)
        

    @property
    def get_cf_ME(self):
        """
        :return: Best fit non-linear least squares curve fit value for envelope mass in solar mass.
        """
        param = "Me"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors

        return self._get_val(param, vdict, edict)

    @property
    def get_cf_VE(self):
        """
        :return: Best fit non-linear least squares curve fit value for velocity of explosion in cm/s.
        """
        try:
            param = "ve"
            vdict = self.sn_obj.CF_fitted_params
            edict = self.sn_obj.CF_fitted_errors
        
            return self._get_val(param, vdict, edict)
        except:
            print('Velocity is not a fitted parameter.')
            return None, None

    @property
    def get_cf_OF(self):
        """
        :return: Best fit non-linear least squares curve fit value for offset from given time of explosion in days.
        """
        param = "Off"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors
        
        return self._get_val(param, vdict, edict)


    def display_curve_fit(self):
        """
        Displays non-linear least squares curve fit results for object.
        :return: Pandas dataframe containing non-linear curve fit results for all parameters.
        """
        store_vals = {}
        columns = [f'{self.sn_obj.model_name} value', f'{self.sn_obj.model_name} error']
        print(f'{self.sn_obj.display_name} model for {self.sn_obj.folder} simple curve fitted values:')
        for key, val in self.sn_obj.CF_fitted_params.items():
            val = val * self.sn_obj.scale[key]
            unit = self.sn_obj.units[key]
            err = self.sn_obj.CF_fitted_errors[key] * self.sn_obj.scale[key]
            print(f'{key}: {val:e} +/- {err:e} {unit}')
            store_vals[f'{key} ({unit})'] = [val, err]
        val_df = pd.DataFrame.from_dict(store_vals, orient='index',
                                        columns=columns)
        return val_df
        # val_df.to_csv(self.make_path(f'{self.model_name}_results.csv'))

    def _analytical_model(self, p, xdata):
        """
        Helper function used in MCMC fitting.
        :param p: parameter list
        :param xdata: times
        :return: Modelled magnitudes at given input parameter values. Returned "modelled" y-data corresponding
        to xdata input.
        """
        args = p
        return self.func(xdata, *args)

    def _loglikelihood_gauss(self, p, xdata, ydata, yerr):
        """
        Helper function used in MCMC sampling. Computes log likelihood of observing input ydata, given parameter
        values p.
        :param p: parameter list
        :param xdata: times
        :param ydata: observed AB magnitudes
        :param yerr: observed error in AB magnitudes
        :return: log-likelihood of observing data given parameters. Assuming a gaussian likelihood function.
        """
        return -0.5 * (np.sum(((ydata - self._analytical_model(p, xdata)) ** 2) / (yerr ** 2) + np.log(
            (2 * np.pi * yerr ** 2) ** 2)))  # we're assuming f = 0

    def _logprior_uniform(self, p):
        """
        Helper function used in MCMC sampling. Set uniform prior for each parameter in p within input bounds.
        :return: 0 if walker is within uniform bounds, -inf if walker leaves uniform bounds.
        """
        if self.use_prev_CF_bounds:

            try:
                fitted = np.array(list(self.sn_obj.CF_fitted_params.values()))
                errors = np.array(list(self.sn_obj.CF_fitted_errors.values()))
                
            except:
                print('Fitted params from simple curve fit undefined. Attempting to curve fit.')

                self.simple_curve_fit()
                fitted = np.array(list(self.sn_obj.CF_fitted_params.values()))
                errors = np.array(list(self.sn_obj.CF_fitted_errors.values()))
                
            self.prior_low = fitted - 5 * errors
            self.prior_high = fitted + 5 * errors

        p = np.array(p)

        # print("parameters = ", p)
        # print("prior_low", self.prior_low)
        # print("prior_high", self.prior_high)
        # print(p > self.prior_low, p < self.prior_high)
        # print(np.all(p > self.prior_low) and np.all(p < self.prior_high))
        if np.all(p > self.prior_low) and np.all(p < self.prior_high):
            # print("yes")
            return 0.0
        return -np.inf

    def _logprob(self, p, xdata, ydata, yerr):
        """
        Helper function used in MCMC sampling. Combines prior and likelihood.
        :param p: parameter list
        :param xdata: times
        :param ydata: observed AB magnitudes
        :param yerr: observed error in AB magnitudes
        :return: log probability of parameters in p given observed data and prior constraints.
        """
        log_prior = self._logprior_uniform(p)
        if np.isfinite(log_prior):
            return log_prior + self._loglikelihood_gauss(p, xdata, ydata, yerr)
        return -np.inf

    def MCMC_fit(self, prior_low: list, prior_high: list, nwalkers=50, nsteps=500, sigma=1, minimize_param_space=False):
        self.use_prev_CF_bounds = minimize_param_space
        self.prior_low = prior_low
        self.prior_high = prior_high
        # INITIALIZATION OF WALKERS
        ndim, nwalkers = self.n_params, nwalkers
        # get simple curve fit values
        fitted, errs = self.simple_curve_fit()

        for i in range(len(fitted)):
            if fitted[i] < self.prior_low[i] or fitted[i] > self.prior_high[i]:
                fitted[i] = errs[i]

                

        pos = fitted + 1e-3 * fitted * np.random.randn(nwalkers, ndim)
        with Pool() as pool:

            sampler = em.EnsembleSampler(nwalkers, ndim, log_prob_fn=self._logprob,
                                        args=(self.xdata_phase, self.ydata_mag, self.yerr_mag), pool=pool)
            sampler.run_mcmc(pos, nsteps=nsteps, progress=True)
        self.samp_chain = sampler.chain
        self.sn_obj.samp_chain = sampler.chain
        self.set_MCMC_bounds_errs(sigma=sigma)
        return self.samp_chain
        

    def set_MCMC_bounds_errs(self, sigma):
        self.sn_obj.MCMC_fitted = {}
        self.sn_obj.MCMC_error = {}
        self.sn_obj.MCMC_sampler = {}
        for i in range(self.n_params):
            param = self.params[i]
            param_arr = self.samp_chain[:, :, i]
            self.sn_obj.MCMC_sampler[param] = param_arr
            self.sn_obj.MCMC_fitted[param] = np.median(param_arr)
            data_within_sig = np.percentile(param_arr, sigma * 34)

            self.sn_obj.MCMC_error[param] = [self.sn_obj.MCMC_fitted[param] - data_within_sig,
                                             self.sn_obj.MCMC_fitted[param] + data_within_sig]

        return self.sn_obj.MCMC_fitted, self.sn_obj.MCMC_error
    

    def save_chain_local(self, local_path):
        locations = []
        for i in range(self.samp_chain.shape[-1]):
            df = pd.DataFrame(self.samp_chain[:, :, i])
            filepath = os.path.join(local_path, f'{self.sn_obj.model_name}_{self.params[i]}_chain.csv')
            locations.append[filepath]
            df.to_csv(filepath)
        print(f'MCMC sampler chains saved at the locations listed here: {locations}')
        return locations

    @property
    def get_MCMC_RE(self):
        """
        :return: Best fit non-linear least squares curve fit value for envelope mass in solar mass.
        """
        param = "Re"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors
        
        return self._get_val(param, vdict, edict)

    @property
    def get_MCMC_ME(self):
        """
        :return: Best fit non-linear least squares curve fit value for envelope mass in solar mass.
        """
        param = "Re"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors
        
        return self._get_val(param, vdict, edict)

    @property
    def get_MCMC_VE(self):
        """
        :return: Best fit non-linear least squares curve fit value for velocity of explosion in cm/s.
        """
        try:
            param = "ve"
            vdict = self.sn_obj.MCMC_fitted_params
            edict = self.sn_obj.MCMC_fitted_errors
        
            return self._get_val(param, vdict, edict)
        except:
            print('Velocity is not a fitted parameter.')
            return None, None

    @property
    def get_MCMC_OF(self):
        """
        :return: Best fit non-linear least squares curve fit value for offset from given time of explosion in days.
        """
        param = "Off"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors
        
        return self._get_val(param, vdict, edict)

