# PLOTTING
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
from matplotlib.patches import Patch
from matplotlib.animation import FFMpegWriter
from shock_cooling_curve.utils import utils
import numpy as np
import corner as cn


class Plotter:
    def __init__(self, sn_obj):
        self.sn_obj = sn_obj
        self.params = list(self.sn_obj.units.keys())
        self.n_params = len(self.params)
        if self.n_params > 3:
            self.MCMC_labels =["$R_e$", "$M_e$", "$v_{shock}$", "$t_{offset}$"]
        else:
            self.MCMC_labels = ["$R_e$", "$M_e$", "$t_{offset}$"]

    def plot_full_curve(self, figsize=(10, 7), errorbar=None, shift=False, show=True,
                        ls='--',
                        fig=None, ax=None):
        '''
        Plots all observed data points (before and after shock cooling ends.)
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.
        '''

        Re, Me, ve, of = self.sn_obj.get_sc_RE()[0], self.sn_obj.get_sc_ME()[0], \
            self.sn_obj.get_sc_VE()[0], self.sn_obj.get_sc_OF()[0]

        if fig is None and ax is None:
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

        legend_elements = []
        for flt in unique_filts:
            # get dataframe for only filter = flt
            filtered = self.sn_obj.data_all[self.sn_obj.data_all[self.sn_obj.flt_colname] == flt]
            # add a column to the `filtered` table; column contains times after computed offset has been subtracted
            filtered['shifted'] = filtered['MJD_S'] - of
            # filter out values that have negative time
            filtered = filtered[filtered['shifted'] > 0]
            # store final times in array `times`
            times = np.array(filtered['shifted'])

            # get the magnitude offset for this filter
            off = offset.pop(0)
            # add the magnitude offset to all magnitude values
            mag_all = np.array(filtered[self.sn_obj.mag_colname]) + off
            if errorbar is not None:
                yerr = np.array(filtered[errorbar])
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

    def plot_given_parameters(self, Re, Me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=False, show=True,
                              ls='--',
                              fig=None, ax=None):

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


    def MCMC_trace(self):
        fig, axs = plt.subplots(self.n_params, 1, figsize=(10, 16))
        for i in range(self.n_params):
            p = self.params[i]
            axs[i].plot(self.sn_obj.MCMC_sampler[p], color='k', alpha=0.5);
            axs[i].set_title(f"Parameter = {p}")
        return fig

    def MCMC_corner(self):
        samples = self.sn_obj.samp_chain[:, 100:, :].reshape((-1, self.n_params))
        fig = cn.corner(
            samples,
            labels = self.MCMC_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12, "loc": "left"})
        fig.suptitle(f"{self.sn_obj.folder} - {self.sn_obj.display_name}")
        return fig

    def make_video(self, ve=0.2, of=0.1):
        '''
        This function will make a video depicting how the shock cooling curve changes for the given dataset
        at discrete values for mass and radius of envelope. It assumes fixed shock velocity and time offset.

        This functionality only exists for the Sapir Waxman (BSG and RSG) models and Piro 2020.

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
                    plt.xlabel('Date (from start of SN) (days)')
                    plt.ylabel("Mag")
                    plt.legend()
                    plt.draw()
                    writer.grab_frame()
                    plt.pause(1.0)
