import numpy as np
import scipy.optimize as opt
import emcee as em
import pandas as pd
import os
import shock_cooling_curve.supernova
from multiprocessing import Pool


class Fitter:
    """Used for fitting a `Supernova` object. 
    Non-linear least squares fitting and MCMC sampling are currently implemented.
    """

    def __init__(self, sn_obj):
        """Initializes a Fitter object to carry out all fitting methods and parameters.
        Instantiates required data and analytical function required for fitting.
        Args:
            sn_obj (Supernova Object): the supernova object that you want to fit
        """
        self.sn_obj = sn_obj
        data = self.sn_obj.reduced_df
        self.xdata_phase = np.array(data[self.sn_obj.shift_date_colname])
        self.ydata_mag = np.array(data[self.sn_obj.rmag_colname])
        self.yerr_mag = np.array(data[self.sn_obj.magerr_colname])

        self.func = self.sn_obj.get_all_mags
        self.params = self.sn_obj.params
        self.n_params = self.sn_obj.n_params

    def nl_curve_fit(self, lower_bounds=None, upper_bounds=None, initial_guess=None):
        """Non-linear least squares curve fitting method. Adopts scipy.optimize.curve_fit

        Args:
            lower_bounds (array-like, optional): n-length array containing lower bounds for n best-fit parameters. Defaults to None.
            upper_bounds (array-like, optional): n-length array containing upper bounds for n best-fit parameters. Defaults to None.
            initial_guess (array-like, optional): n-length array containing initial guess for n best-fit parameters. Defaults to None.

        Returns:
            tuple: tuple of arrays containing best fit parameters and 1-sigma standard deviation errors.
        """
        if lower_bounds is not None:
            print('lower_bounds have been provided. These will be used for curve fitting.')
            self.sn_obj.lower_bounds = lower_bounds
        else:
            print('No lower bounds were provided. Here are the lower bounds we used:')
            print(f'Lower bounds = {self.sn_obj.lower_bounds}.')

        if upper_bounds is not None:
            print('upper_bounds have been provided. These will be used for curve fitting.')
            self.sn_obj.upper_bounds = upper_bounds
        else:
            print('No upper bounds were provided. Here are the upper bounds we used:')
            print(f'Upper bounds = {self.sn_obj.upper_bounds}.')

        if initial_guess is not None:
            print('initial_guess have been provided. These will be used for curve fitting')
            self.sn_obj.initial = initial_guess
        else:
            print('initial_guess not provided. We use this initial guess:')
            print(f'Initial Guess = {self.sn_obj.initial}.')

        popt, pcov = opt.curve_fit(
            f=self.func,
            xdata=self.xdata_phase,
            ydata=self.ydata_mag,
            p0=self.sn_obj.initial,
            sigma=self.yerr_mag,
            bounds=(self.sn_obj.lower_bounds, self.sn_obj.upper_bounds)
        )
        sigma = np.sqrt(np.diag(pcov))

        self.sn_obj.CF_fitted_params = dict(zip(self.params, popt))
        self.sn_obj.CF_fitted_errors = dict(zip(self.params, sigma))

        return popt, sigma

    def minimize(self, initial_guess=None):
        """Applies scipy.optimize.minimize. No bounds are taken.

        Args:
            initial_guess (array-like, optional): initial guess for parameters. Defaults to None.

        Returns:
            OptimizeResult: attributes as x = solution array; success = bool; message = cause for termination
        """
        if initial_guess is not None:
            print('initial_guess have been provided.')
            self.sn_obj.initial = initial_guess
        else:
            print('initial_guess not provided. We use this initial guess:')
            print(f'Initial Guess = {self.sn_obj.initial}.')
        bounds = []
        for i in range(len(self.sn_obj.lower_bounds)):
            bounds.append([self.sn_obj.lower_bounds[i], self.sn_obj.upper_bounds[i]])

        min_func = lambda *args: self._loglikelihood_gauss(*args)

        return opt.minimize(min_func, x0=self.sn_obj.initial, method='L-BFGS-B',
                            args=(self.xdata_phase, self.ydata_mag, self.yerr_mag), bounds=bounds, options={'maxiter':10000})

    def _get_val(self, param, value_dict, err_dict):
        """helper function to get one particular scaled value

        Args:
            param (str): param of interest: 're', 'me', 've', or 'off'
            value_dict (dict): dictionary containing best fit values
            err_dict (dict): dictionary containing 1-sigma error on best fit values

        Returns:
            tuple: scaled best fit value and error
        """
        print(f'Units: {self.sn_obj.units[param]}')
        return value_dict[param] * self.sn_obj.scale[param], err_dict[param] * self.sn_obj.scale[param]

    @property
    def cf_RE(self):
        """Best fit non-linear least squares curve fit value for envelope radius in solar radii.

        Returns:
            tuple: tuple of best fit radius and error
        """
        param = "re"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors

        return self._get_val(param, vdict, edict)

    @property
    def cf_ME(self):
        """Best fit non-linear least squares curve fit value for envelope mass in solar masses.

        Returns:
            tuple: tuple of best fit mass and error
        """

        param = "me"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors

        return self._get_val(param, vdict, edict)

    @property
    def cf_VE(self):
        """Best fit non-linear least squares curve fit value for shock velocity in 10^9 cm/s.

        Returns:
            tuple: tuple of best fit shock velocity and error
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
    def cf_OF(self):
        """Best fit non-linear least squares curve fit value for time offset from start of supernova in days.

        Returns:
            tuple: tuple of best fit time offset and error
        """
        param = "Off"
        vdict = self.sn_obj.CF_fitted_params
        edict = self.sn_obj.CF_fitted_errors

        return self._get_val(param, vdict, edict)

    def display_results(self, method='curvefit'):
        """Displays non-linear least squares curve fit results for objects.

        Args:
            method (str, optional): 'curvefit' or 'MCMC'. Defaults to 'curvefit'.

        Returns:
            pandas DataFrame: table containing non-linear curve fit results for all parameters.
        """
        if method == 'curvefit':
            print('Model used: non linear least squared fitting')
            return self.__display_results_helper(value_dict=self.sn_obj.CF_fitted_params,
                                                 err_dict=self.sn_obj.CF_fitted_errors)
        elif method == 'MCMC':
            print('Model used: MCMC Sampling')
            return self.__display_results_helper(value_dict=self.sn_obj.MCMC_fitted_params,
                                                 err_dict=self.sn_obj.MCMC_fitted_errors)
        else:
            print("Sorry! That model has not been implemented yet. The options are 'curvefit' or 'MCMC'")

    def __display_results_helper(self, value_dict, err_dict):
        """Helper function to display results.

        Args:
            value_dict (dict): dictionary containing best fit values
            err_dict (dict): dictionary containing 1-sigma error on best fit values

        Returns:
            pandas DataFrame: table containing best fit values and errors
        """
        store_vals = {}
        columns = [f'{self.sn_obj.model_name} value', f'{self.sn_obj.model_name} error']
        print(f'{self.sn_obj.display_name} model - {self.sn_obj.objname}:')
        for key, val in value_dict.items():
            val = val * self.sn_obj.scale[key]
            unit = self.sn_obj.units[key]
            err = np.array(err_dict[key]) * self.sn_obj.scale[key]
            print(f'{key}: {val} +/- {err} {unit}')
            store_vals[f'{key} ({unit})'] = [val, err]
        val_df = pd.DataFrame.from_dict(store_vals, orient='index',
                                        columns=columns)

        return val_df

    def _analytical_model(self, p, xdata):
        """Helper function used in MCMC fitting.
        Args:
            p (list): list of parameter guesses
            xdata (array-like): time data

        Returns:
            array-like: Modelled magnitudes at given input parameter values. Returned "modelled" y-data corresponding to xdata input.
        """
        args = p
        return self.func(xdata, *args)

    def _loglikelihood_gauss(self, p, xdata, ydata, yerr):
        """Helper function used in MCMC sampling. Computes log likelihood of observing input ydata, given parameter
        values p.
        Args:
            p (list): list of parameter guesses
            xdata (array-like): time data
            ydata (array-like): observed AB magnitudes (extinction applied)
            yerr (array-like): observed error in AB magnitudes

        Returns:
            array-like: log-likelihood of observing data given parameters. Assuming a gaussian likelihood function.
        """
        return -0.5 * (np.sum(((ydata - self._analytical_model(p, xdata)) ** 2) / (yerr ** 2) + np.log(
            (2 * np.pi * yerr ** 2) ** 2)))  # we're assuming f = 0

    def _logprior_uniform(self, p):
        """Helper function used in MCMC sampling. Set uniform prior for each parameter in p within input bounds.
        Args:
            p (list): list of parameter guesses
        
        Returns:
            float: 0 if walker is within uniform bounds, -inf if walker leaves uniform bounds.
        """

        p = np.array(p)

        # print("parameters = ", p)
        # print("prior_low", self.prior_low)
        # print("prior_high", self.prior_high)
        # print(p > self.prior_low, p < self.prior_high)
        # print(np.all(p > self.prior_low) and np.all(p < self.prior_high))
        if np.all(p > self.prior_low) and np.all(p < self.prior_high):
            return 0.0
        return -np.inf

    def _logprob(self, p, xdata, ydata, yerr):
        """Helper function used in MCMC sampling. Combines prior and likelihood.
        Args:
            p (list): list of parameter guesses
            xdata (array-like): time data
            ydata (array-like): observed AB magnitudes (extinction applied)
            yerr (array-like): observed error in AB magnitudes

        Returns:
            array-like: log probability of parameters in p given observed data and prior constraints.
        """
        log_prior = self._logprior_uniform(p)
        if np.isfinite(log_prior):
            return log_prior + self._loglikelihood_gauss(p, xdata, ydata, yerr)
        return -np.inf

    def MCMC_fit(self,
                 prior_low: list,
                 prior_high: list,
                 nwalkers=50,
                 nsteps=500,
                 sigma=1,
                 use_initial_params=None,
                 initialize_using_CF=True,
                 minimize_param_space=False,
                 burnin=0):
        """Carries out MCMC Sampling to shock cooling model to observed data.

        Args:
            prior_low (list): uniform prior lower bounds for best fit parameters
            prior_high (list): uniform prior upper bounds for best fit parameters
            nwalkers (int, optional): number of MCMC walkers. Defaults to 50.
            nsteps (int, optional): number of steps MCMC walkers should traverse to reach a steady state solution. Defaults to 500.
            sigma (int, optional): confidence interval bound limit. Defaults to 1-sigma confidence.
            use_initial_params (list, optional): list containing initialization point of walkers for each parameter. Defaults to None.
            initialize_using_CF (bool, optional): if True, walkers will be initialized at curve fit results. Defaults to True.
            minimize_param_space (bool, optional): if True, walkers will be bound to a 5-sigma space around curve fit results. Defaults to False.

        Returns:
            n-d array: MCMC sampler chain containing results for all parameters.
        """
        self.use_prev_CF_bounds = minimize_param_space
        self.prior_low = prior_low
        self.prior_high = prior_high
        # INITIALIZATION OF WALKERS
        ndim, nwalkers = self.n_params, nwalkers
        # get simple curve fit values

        if use_initial_params is not None:

            assert np.all(use_initial_params > self.prior_low) and np.all(
                use_initial_params < self.prior_high), "The initial values you have provided are out of the uniform prior's bounds."
            use_initial_params = np.array(use_initial_params)
            pos = use_initial_params + 1e-3 * use_initial_params * np.random.randn(nwalkers, ndim)

        else:
            print("Initial values were not provided. Using results from curve fitting as initial parameters.")
            try:
                fitted, errs = np.array(list(self.sn_obj.CF_fitted_params.values())), np.array(
                    list(self.sn_obj.CF_fitted_errors.values()))
            except:
                fitted, errs = self.nl_curve_fit()

            # make sure that the initial parameters obtained using curve fit are within prior bounds
            # if they are not, use the error instead (seems to be well behaved if param is not)
            for i in range(len(fitted)):
                if fitted[i] < self.prior_low[i] or fitted[i] > self.prior_high[i]:
                    fitted[i] = errs[i]

            if self.use_prev_CF_bounds:
                # reduces prior bounds to just       
                self.prior_low = fitted - 5 * errs
                self.prior_high = fitted + 5 * errs

                print('Priors have been adjust to a smaller uniform parameter space.')
                print(f'Updated prior_low = {self.prior_low}')
                print(f'Updated prior_high = {self.prior_high}')

            if initialize_using_CF:
                pos = fitted + 1e-3 * fitted * np.random.randn(nwalkers, ndim)

        assert len(self.prior_high) >= 3, "prior_high length < 3."
        assert len(self.prior_low) >= 3, "prior_low length < 3."
        assert pos.shape == (nwalkers, ndim), f"pos.shape = {pos.shape}, but should be {(nwalkers, ndim)}."

        with Pool() as pool:
            sampler = em.EnsembleSampler(nwalkers, ndim, log_prob_fn=self._logprob,
                                         args=(self.xdata_phase, self.ydata_mag, self.yerr_mag), pool=pool)
            sampler.run_mcmc(pos, nsteps=nsteps, progress=True)
        self.samp_chain = sampler.chain
        self.sn_obj.samp_chain = sampler.chain
        self.set_MCMC_bounds_errs(sigma=sigma, burnin=burnin)
        return self.samp_chain

    def set_MCMC_bounds_errs(self, sigma, burnin=0):
        """Resets MCMC best fit results according to sigma-level provided and burnin.

        Args:
            sigma (int): error bound sigma level
            burnin (int, optional): number of walkers to neglect before computing best fit parameter. Defaults to 0.

        Returns:
            tuple: tuple of dicts containing best fit parameters and error bounds for each parameter.
        """
        self.sn_obj.MCMC_fitted_params = {}
        self.sn_obj.MCMC_fitted_errors = {}
        self.sn_obj.MCMC_sampler = {}
        for i in range(self.n_params):
            param = self.params[i]
            param_arr = self.samp_chain[:, burnin:, i]
            self.sn_obj.MCMC_sampler[param] = param_arr
            self.sn_obj.MCMC_fitted_params[param] = np.median(param_arr)
            data_within_sig = np.percentile(param_arr, sigma * 34)

            self.sn_obj.MCMC_fitted_errors[param] = [self.sn_obj.MCMC_fitted_params[param] - data_within_sig,
                                                     self.sn_obj.MCMC_fitted_params[param] + data_within_sig]

        return self.sn_obj.MCMC_fitted_params, self.sn_obj.MCMC_fitted_errors

    def save_chain_local(self, local_path="", burnin=0):
        """Saves MCMC chains to local path in separate csv files for each parameter.

        Args:
            local_path (str, optional): location where you want to store the MCMC chains. 
                Defaults to current working directory.
            burnin (int, optional): number of steps to remove before saving the chain.
                Defaults to 0.

        Returns:
            list: locations of saved MCMC chains
        """
        locations = []
        for i in range(self.samp_chain.shape[-1]):
            df = pd.DataFrame(self.samp_chain[:, burnin:, i])
            filepath = os.path.join(local_path, f'{self.sn_obj.objname}_{self.sn_obj.model_name}_{self.params[i]}_chain.csv')
            locations.append(filepath)
            df.to_csv(filepath)
        print(f'MCMC sampler chains saved at the locations listed here: {locations}')
        return locations

    @property
    def MCMC_RE(self):
        """Best fit result for progenitor envelope radius computed by MCMC

        Returns:
            tuple: best fit radius and uncertainty
        """
        param = "re"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors

        return self._get_val(param, vdict, edict)

    @property
    def MCMC_ME(self):
        """Best fit result for progenitor envelope mass computed by MCMC

        Returns:
            tuple: best fit mass and uncertainty
        """
        param = "me"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors

        return self._get_val(param, vdict, edict)

    @property
    def MCMC_VE(self):
        """Best fit result for shock velocity computed by MCMC

        Returns:
            tuple: best fit shock velocity and uncertainty
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
    def MCMC_OF(self):
        """Best fit result for time offset from start of SN computed by MCMC

        Returns:
            tuple: best fit time offset and uncertainty
        """
        param = "Off"
        vdict = self.sn_obj.MCMC_fitted_params
        edict = self.sn_obj.MCMC_fitted_errors

        return self._get_val(param, vdict, edict)
