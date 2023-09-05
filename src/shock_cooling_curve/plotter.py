# PLOTTING
from shock_cooling_curve.utils import utils
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
from matplotlib.patches import Patch
from matplotlib.animation import FFMpegWriter
import numpy as np
import corner as cn


class Plotter:
    """
    Plotting modules. Plots best fit results obtained from MCMC sampling or curve fitting.
    """

    def __init__(self, sn_obj):
        self.sn_obj = sn_obj
        self.params = self.sn_obj.params
        self.n_params = self.sn_obj.n_params
        self.objname = self.sn_obj.objname
        self.model_name = self.sn_obj.display_name
        if self.n_params > 3:
            self.MCMC_labels = ["$R_e$", "$M_e$", "$v_{shock}$", "$t_{offset}$"]

        else:
            self.MCMC_labels = ["$R_e$", "$M_e$", "$t_{offset}$"]
        self.display_params = dict(zip(self.params, self.MCMC_labels))

    def _get_discrete_values(self, df, flt, of, errorbar_col):
        filtered = df[df[self.sn_obj.flt_colname] == flt].copy()
        filtered['shifted'] = filtered['MJD_S'] - of
        filtered = filtered[filtered['shifted'] > 0]
        times = np.array(filtered['shifted'])
        mag_all = np.array(filtered[self.sn_obj.mag_colname])
        if errorbar_col is not None:
            mag_err = np.array(filtered[errorbar_col])
            return times, mag_all, mag_err
        else:
            return times, mag_all, np.zeros(len(mag_all))
    
    def _plot_helper(self, df, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True, show=True,
                              ls='--',
                              fig=None, ax=None, legend =True):
        
        if fig == None and ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        unique_filts = self.sn_obj.get_filts()
        n = len(unique_filts)
        minmag = min(df[self.sn_obj.mag_colname]) - n
        maxmag = max(df[self.sn_obj.mag_colname]) + n

        # only get times within shock cooling time range
        t = np.linspace(0.01, max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]) + 1, 100)

        yerr = errorbar
        if shift:
            offset = utils.build_offset(n)
        else:
            offset = [0] * len(unique_filts)

        legend_elements = []
        for flt in unique_filts:
            times, mag_all, mag_err = self._get_discrete_values(df, flt, of,
                                                    errorbar_col= errorbar)
            off = offset.pop(0)
            mag_all += off

            # DISCRETE
            ax.errorbar(x=times, y=mag_all, yerr=mag_err, fmt='o', markersize=14, markeredgecolor='k',
                        color=utils.get_mapping('flt', flt, 'Color'))

            legend_elements.append(Patch(facecolor=utils.get_mapping('flt', flt, 'Color'),
                                         label=utils.get_label(flt, off)))
            # CONTINUOUS
            vals = self.sn_obj._mags_per_filter(t, filtName=flt, Re = Re, Me = Me, ve = ve) + off

            ax.plot(t, vals, linestyle=ls, color=utils.get_mapping('flt', flt, 'Color'))
            if legend:
                ax.legend(handles=legend_elements, frameon=False, ncol=2)

            ax.set_title(f'{self.objname} + {self.model_name}')
            ax.set_xlabel('Phase (d)')
            ax.set_ylabel('Apparent Mag (m)')
        ax.invert_yaxis()
        if show:
            plt.show()
        return fig
        

    def plot_full_curve(self, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True, show=True,
                              ls='--',
                              fig=None, ax=None):
        '''
        Plots all observed data points (before and after shock cooling ends.)
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.
        '''
        return self._plot_helper(df = self.sn_obj.data_all, Re=Re, Me=Me, ve=ve, of=of, figsize=figsize, errorbar=errorbar, shift=shift, show=show,
                        ls=ls,
                        fig=fig, ax=ax )
    
    def plot_given_observed_data(self, data, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True, show=True,
                              ls='--',
                              fig=None, ax=None):
        return self._plot_helper(df = data, Re=Re, Me=Me, ve=ve, of=of, figsize=figsize, errorbar=errorbar, shift=shift, show=show,
                        ls=ls,
                        fig=fig, ax=ax )

      

    def plot_given_parameters(self, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True, show=True,
                              ls='--',
                              fig=None, ax=None):
        '''
        Plots all observed data points before shock cooling ends.
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.
        '''
        return self._plot_helper(df = self.sn_obj.reduced_df, Re=Re, Me=Me, ve=ve, of=of, figsize=figsize, errorbar=errorbar, shift=shift, show=show,
                        ls=ls,
                        fig=fig, ax=ax)



    def MCMC_trace(self, burnin=0, color=False):
        """
        Generates the sampler trace plots for all the walkers used in MCMC sampling.
        """
        fig, axs = plt.subplots(self.n_params, 1, figsize=(10, 16))
        for i in range(self.n_params):
            p = self.params[i]
            if color:
                axs[i].plot(np.array(self.sn_obj.MCMC_sampler[p][:, burnin:]).T, alpha=0.5);
            else:
                axs[i].plot(np.array(self.sn_obj.MCMC_sampler[p][:, burnin:]).T, color='k', alpha=0.5);

            axs[i].set_title(f"{self.display_params[p]}")
        plt.suptitle(f'{self.objname} + {self.model_name} parameter sampler chains')
        
        #fig.suptitle(f"{self.sn_obj.objname} - {self.sn_obj.display_name}")

        return fig

    def MCMC_corner(self, burnin=0):
        """
        Generates corner plot for the posterior distributions of parameter values.
        """
        samples = self.sn_obj.samp_chain[:, burnin:, :].reshape((-1, self.n_params))
        fig = cn.corner(
            samples,
            labels=self.MCMC_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12, "loc": "left"})
        fig.suptitle(f"{self.sn_obj.objname} - {self.sn_obj.display_name}")
        return fig

    def make_video(self, ve=0.2, of=0.1):
        '''
        This function will make a video depicting how the shock cooling curve changes for the given dataset
        at discrete values for mass and radius of envelope. It assumes fixed shock velocity and time offset.

        This functionality only exists for the Sapir Waxman (BSG and RSG) models and Piro 2020.
        :param ve: (Optional) Velocity of shock moving outwards - fixed at input value.
        :param of: (Optional) Time offset (in days) from SN explosion - fixed at input value.
        :return:
        '''

        metadata = dict(title='Shock Cooling', artist='Matplotlib', comment='Shock Cooling Fit with varying me and re')
        writer = FFMpegWriter(fps=4, metadata=metadata)
        phase_combo = np.array(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname])
        mags = self.sn_obj.reduced_df['MAG']
        fig = plt.figure(figsize=(8, 7))
        # plt.gca().invert_yaxis()
        sr = np.arange(0, max(phase_combo))
        with writer.saving(fig, f"{self.sn_obj.path_to_storage} + {self.sn_obj.model_name}.mp4", dpi=200):
            for r in np.linspace(0.01, 10, 20):
                for m in [0.1, 0.3, 0.5, 0.7]:
                    plt.clf()
                    plt.gca().invert_yaxis()
                    test_values = self.sn_obj.get_all_mags(times=phase_combo, Re=r, Me=m, ve=ve, of=of)
                    test_sr = self.sn_obj.get_all_mags(times=sr, Re=r, Me=m, ve=ve, of=of)
                    plt.scatter(phase_combo, mags, label='original')
                    plt.plot(sr, test_sr, label='continuous')
                    plt.scatter(phase_combo, test_values, label='fitted')
                    plt.title(f're = {r}, me={m}, RMSE = {np.round(np.sqrt(np.mean((test_values - mags) ** 2)), 2)}')
                    plt.xlabel(' (from start of SN) (days)')
                    plt.ylabel("Mag")
                    plt.legend()
                    plt.draw()
                    writer.grab_frame()
                    plt.pause(1.0)
