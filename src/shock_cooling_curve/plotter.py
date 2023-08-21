# PLOTTING
from src.shock_cooling_curve.utils import utils
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
        if self.n_params > 3:
            self.MCMC_labels = ["$R_e$", "$M_e$", "$v_{shock}$", "$t_{offset}$"]

        else:
            self.MCMC_labels = ["$R_e$", "$M_e$", "$t_{offset}$"]
        self.display_params = dict(zip(self.params, self.MCMC_labels))

    def _get_discrete_values(self, df, flt, of, errorbar_col, full=False):
        if full:
            filtered = df[df[self.sn_obj.flt_colname] == flt]
            filtered['shifted'] = filtered['MJD_S'] - of
        filtered = filtered[filtered['shifted'] > 0]
        times = np.array(filtered['shifted'])
        mag_all = np.array(filtered[self.sn_obj.mag_colname])
        if errorbar_col is not None:
            mag_err = np.array(filtered[errorbar_col])
            return times, mag_all, mag_err
        else:
            return times, mag_all, np.zeros(len(mag_all))

    def _get_full_values(self):
        return

    def plot_full_curve(self, figsize=(10, 7), errorbar_column=None, shift=False, show=True,
                        ls='--',
                        fig=None, ax=None):
        '''
        Plots all observed data points (before and after shock cooling ends.)
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.
        '''

        fitted, _ = self.sn_obj.get_curvefit_values()

        Re, Me, ve, of = fitted

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        unique_filts = self.sn_obj.get_filts()
        n = len(unique_filts)
        minmag = min(self.sn_obj.reduced_df[self.sn_obj.mag_colname]) - n
        maxmag = max(self.sn_obj.reduced_df[self.sn_obj.mag_colname]) + n
        t = np.linspace(0.01, max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]) + 1, 100)

        yerr = errorbar_column
        if shift:
            offset = utils.build_offset(n)
        else:
            offset = [0] * len(unique_filts)

        legend_elements = []
        for flt in unique_filts:
            times, mag_all, mag_err = self._get_discrete_values(self, self.sn_obj.data_all, flt, of,
                                                                errorbar_col= errorbar_column, full=False)
            off = offset.pop(0)
            mag_all += off

            # DISCRETE
            ax.errorbar(x=times, y=mag_all, yerr=yerr, fmt='o', markersize=14, markeredgecolor='k',
                        color=utils.get_mapping('flt', flt, 'Color'))

            legend_elements.append(Patch(facecolor=utils.get_mapping('flt', flt, 'Color'),
                                         label=utils.get_label(flt, off)))
            # CONTINUOUS
            vals = self.sn_obj.mags_per_filter(t, Re, Me, ve, filtName=flt) + off
            ax.plot(t, vals, linestyle=ls, color=utils.get_mapping('flt', flt, 'Color'))
            ax.legend(handles=legend_elements, frameon=False, ncol=2)

            ax.set_title(f'{self.sn_obj.filename[:-9]} + {self.sn_obj.display_name}')
            ax.set_xlabel('Phase (d)')
            ax.set_ylabel('Apparent Mag (m)')
        ax.invert_yaxis()
        if show:
            plt.show()
        return fig

    def plot_given_parameters(self, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True, show=True,
                              ls='--',
                              fig=None, ax=None):
        """
        
        """

        if fig == None and ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        unique_filts = self.sn_obj.get_filts()
        n = len(unique_filts)
        minmag = min(self.sn_obj.reduced_df[self.sn_obj.mag_colname]) - n
        maxmag = max(self.sn_obj.reduced_df[self.sn_obj.mag_colname]) + n
        t = np.linspace(0.01, max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]) + 1, 100)

        yerr = errorbar
        if shift:
            offset = utils.build_offset(n)
        else:
            offset = [0] * len(unique_filts)

        # HARDCODED
        # print('unique_filts', unique_filts)
        # for flt in self.sn_obj.filt_to_wvl.keys():
        legend_elements = []
        for flt in unique_filts:
            filtered = self.sn_obj.reduced_df[self.sn_obj.reduced_df[self.sn_obj.flt_colname] == flt]
            times = np.array(filtered['MJD_S'])

            # filter negative phases

            off = offset.pop(0)
            mag_all = np.array(filtered[self.sn_obj.mag_colname]) + off
            if errorbar != None:
                yerr = np.array(filtered[errorbar])
            # DISCRETE
            ax.errorbar(x=times, y=mag_all, yerr=yerr, fmt='o', markersize=14, markeredgecolor='k',
                        color=utils.get_mapping('flt', flt, 'Color'))

            legend_elements.append(Patch(facecolor=utils.get_mapping('flt', flt, 'Color'),
                                         label=utils.get_label(flt, off)))
            # CONTINUOUS
            vals = self.sn_obj._mags_per_filter(t,filtName=flt, Re = Re, Me = Me, ve = ve) + off

            ax.plot(t, vals, linestyle=ls, color=utils.get_mapping('flt', flt, 'Color'))
            ax.legend(handles=legend_elements, frameon=False, ncol=2)

            ax.set_title(f'{self.sn_obj.filename[:-9]} + {self.sn_obj.display_name}')
            ax.set_xlabel('Phase (d)')
            ax.set_ylabel('Apparent Mag (m)')
        ax.invert_yaxis()
        if show:
            plt.show()
        return fig

    def MCMC_trace(self, burnin=0):
        """
        Generates the sampler trace plots for all the walkers used in MCMC sampling.
        """
        fig, axs = plt.subplots(self.n_params, 1, figsize=(10, 16))
        for i in range(self.n_params):
            p = self.params[i]
            axs[i].plot(np.array(self.sn_obj.MCMC_sampler[p][:, burnin:]).T, color='k', alpha=0.5);
            axs[i].set_title(f"{self.display_params[p]}")
        
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
        #fig.suptitle(f"{self.sn_obj.objname} - {self.sn_obj.display_name}")
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