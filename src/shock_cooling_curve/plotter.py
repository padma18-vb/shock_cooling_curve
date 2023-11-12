# PLOTTING
from shock_cooling_curve.utils import utils
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
from matplotlib.patches import Patch
from matplotlib.animation import FFMpegWriter
import numpy as np
import corner as cn


class Plotter:
    """Plotting modules. Plots best fit results obtained from MCMC sampling or curve fitting.
    """

    def __init__(self, sn_obj):
        """Initializes a Plotter. Contains information required for plotting a light curve.

        Args:
            sn_obj (Supernova Object): the supernova object that you want to fit
        """
        self.sn_obj = sn_obj
        try:
            self.params = self.sn_obj.params
            self.n_params = self.sn_obj.n_params
        except:
            print('There are no best fit results for this object. To compute',
                  'this use the Fitter object on ', sn_obj)
        self.objname = self.sn_obj.objname
        self.model_name = self.sn_obj.display_name
        if self.n_params > 3:
            self.MCMC_labels = ["$R_e / R_\odot\ $", "$M_e / M_\odot\ $", "$v_{shock} (10^9 cm/s) $", "$t_{offset}$ (days)"]

        else:
            self.MCMC_labels = ["$R_e / R_\odot\ $", "$M_e / M_\odot\ $",  "$t_{offset}$ (days)"]
        self.display_params = dict(zip(self.params, self.MCMC_labels))
        self.synthetic_phot = {}

    def get_synthetic_phot(self):
        """Gets synthetic photometry values (offset by a integer for every band)

        Returns:
            dict: key: filter; value: sythetic photometry values for filter
        """
        return self.synthetic_phot
    def _get_discrete_values(self, df, flt, errorbar_col):
        """Helper function used to obtain discrete values

        Args:
            df (pd.DataFrame): table containing observed data
            flt (str): specific observing filter
            errorbar_col (str): name of column containing observed photometry error or None.

        Returns:
            tuple: phase data, observed mags, observed magnitude error
        """
        filtered = df[df[self.sn_obj.flt_colname] == flt].copy()
        #dates
        filtered['shifted'] = filtered['MJD_S']

        filtered = filtered[filtered['shifted'] > 0]
        times = np.array(filtered['shifted'])
        # photometry
        mag_all = np.array(filtered[self.sn_obj.rmag_colname])
        
        if errorbar_col is not None:
            mag_err = np.array(filtered[errorbar_col])
            return times, mag_all, mag_err
        else:
            return times, mag_all, np.zeros(len(mag_all))

    def _plot_helper(self, df: object, re: float, me: float, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True,
                     ls='-',
                     fig=None, ax=None, legend=True, legend_loc='best'):
        """Helper function used to plot lightcurve in all bands.

        Args:
            df (pd.DataFrame): table containing observed data
            re (float): _description_
            me (float): _description_
            ve (float, optional): _description_. Defaults to None.
            of (int, optional): _description_. Defaults to 0.
            figsize (tuple, optional): _description_. Defaults to (7, 7).
            errorbar (str, optional): name of column containing observed photometry error. Defaults to None.
            shift (bool, optional): True if all photometry should be grouped by filter and shifted. Defaults to True.
            ls (str, optional): linestyle used for synthetic photometry curve. Defaults to '-'.
            fig (matplotlib.Figure, optional): figure used to plot. Defaults to None.
            ax (matplotlib.Axes, optional): axes used to plot. Defaults to None.
            legend (bool, optional): If True, legend is shown. Defaults to True.
            legend_loc (str or tuple, optional): Any input to matplotlib.legend.loc 
                (can be tuple of coordinates or string describing position). Defaults to 'best'.

        Returns:
            tuple: fig, ax, legend elements
        """

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        unique_filts = self.sn_obj.get_filts()
        n = len(unique_filts)

        # only get times within shock cooling time range
        t = np.linspace(0.001,
                        max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]) + 1, 100)

        if shift:
            offset = utils.build_offset(n)
        else:
            offset = [0] * len(unique_filts)

        legend_elements = []
        for flt in unique_filts:
            times, mag_all, mag_err = self._get_discrete_values(df, flt, errorbar_col=errorbar)
            off = offset.pop(0)
            mag_all += off
            times -= of

            # DISCRETE

            ax.errorbar(x=times, y=mag_all, yerr=mag_err, fmt='o', markersize=14, markeredgecolor='k', capsize=10,
                        color=utils.get_mapping('flt', flt, 'Color'))

            legend_elements.append(Patch(facecolor=utils.get_mapping('flt', flt, 'Color'),
                                         label=utils.get_label(flt, off)))
            
            # CONTINUOUS
            vals = self.sn_obj._mags_per_filter(t, filtName=flt, re=re, me=me, ve=ve) + off
            self.synthetic_phot[flt] = vals

            cond = np.where(t <= max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]))
            cond_false = np.where(t > max(self.sn_obj.reduced_df[self.sn_obj.shift_date_colname]))
            ax.plot(t[cond], vals[cond], linestyle=ls, color=utils.get_mapping('flt', flt, 'Color'))
            ax.plot(t[cond_false], vals[cond_false], linestyle='--', color=utils.get_mapping('flt', flt, 'Color'))

            ax.set_title(f'{self.objname} + {self.model_name}')
            ax.set_xlabel('Time from Explosion (days)')
            ax.set_ylabel('Apparent Magnitude (mag)')

            ax.legend(handles=legend_elements, frameon=False, ncol=2, loc=legend_loc)

        ax.invert_yaxis()
        if not legend:
            leg = ax.get_legend()
            leg.set_visible(False)
            return fig, ax, legend_elements
        else:
            return fig, ax

    def plot_full_curve(self, re, me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True,
                        ls='--',
                        fig=None, ax=None, legend=True, legend_loc='best'):
        """Plots all observed data points (before and after shock cooling ends.)
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.

        Args:
            df (pd.DataFrame): table containing observed data
            re (float): radius of extended material
            me (float): mass of extended material
            ve (float, optional): shock velocity. Defaults to None.
            of (int, optional): time offset from start of SN. Defaults to 0.
            figsize (tuple, optional): matplotlib figure size. Defaults to (7, 7).
            errorbar (str, optional): name of column containing observed photometry error. Defaults to None.
            shift (bool, optional): True if all photometry should be grouped by filter and shifted. Defaults to True.
            ls (str, optional): linestyle used for synthetic photometry curve. Defaults to '-'.
            fig (matplotlib.Figure, optional): figure used to plot. Defaults to None.
            ax (matplotlib.Axes, optional): axes used to plot. Defaults to None.
            legend (bool, optional): If True, legend is shown. Defaults to True.
            legend_loc (str or tuple, optional): Any input to matplotlib.legend.loc 
                (can be tuple of coordinates or string describing position). Defaults to 'best'.

        Returns:
            tuple: fig, ax, legend elements
        """
        return self._plot_helper(df=self.sn_obj.data_all, re=re, me=me, ve=ve, of=of,
                                 figsize=figsize, errorbar=errorbar,
                                 shift=shift, ls=ls,
                                 fig=fig, ax=ax, legend=legend, legend_loc=legend_loc)

    def plot_given_observed_data(self, data, re, me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True,
                                 ls='--', fig=None, ax=None, legend=True, legend_loc='best'):
        '''
        Plots given observed data points.
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows given observed data points and the fitted shock cooling curve (to shock cooling data).

        Args:
            df (pd.DataFrame): table containing observed data
            re (float): radius of extended material
            me (float): mass of extended material
            ve (float, optional): shock velocity. Defaults to None.
            of (int, optional): time offset from start of SN. Defaults to 0.
            figsize (tuple, optional): matplotlib figure size. Defaults to (7, 7).
            errorbar (str, optional): name of column containing observed photometry error. Defaults to None.
            shift (bool, optional): True if all photometry should be grouped by filter and shifted. Defaults to True.
            ls (str, optional): linestyle used for synthetic photometry curve. Defaults to '-'.
            fig (matplotlib.Figure, optional): figure used to plot. Defaults to None.
            ax (matplotlib.Axes, optional): axes used to plot. Defaults to None.
            legend (bool, optional): If True, legend is shown. Defaults to True.
            legend_loc (str or tuple, optional): Any input to matplotlib.legend.loc 
                (can be tuple of coordinates or string describing position). Defaults to 'best'.

        Returns:
            tuple: fig, ax, legend elements
        '''
        return self._plot_helper(df=data, re=re, me=me, ve=ve, of=of,
                                 figsize=figsize, errorbar=errorbar,
                                 shift=shift, ls=ls,
                                 fig=fig, ax=ax, legend=legend, legend_loc=legend_loc)

    def plot_given_parameters(self, re, me, ve=None, of=0, figsize=(7, 7), errorbar=None, shift=True,
                              ls='--',
                              fig=None, legend=False, legend_loc=None, ax=None):
        '''
        Plots all observed data points before shock cooling ends.
        Computes fitted magnitudes for times before end of shock cooling.
        Plot shows all observed data points and the fitted shock cooling curve.

        Args:
            df (pd.DataFrame): table containing observed data
            re (float): radius of extended material
            me (float): mass of extended material
            ve (float, optional): shock velocity. Defaults to None.
            of (int, optional): time offset from start of SN. Defaults to 0.
            figsize (tuple, optional): matplotlib figure size. Defaults to (7, 7).
            errorbar (str, optional): name of column containing observed photometry error. Defaults to None.
            shift (bool, optional): True if all photometry should be grouped by filter and shifted. Defaults to True.
            ls (str, optional): linestyle used for synthetic photometry curve. Defaults to '-'.
            fig (matplotlib.Figure, optional): figure used to plot. Defaults to None.
            ax (matplotlib.Axes, optional): axes used to plot. Defaults to None.
            legend (bool, optional): If True, legend is shown. Defaults to True.
            legend_loc (str or tuple, optional): Any input to matplotlib.legend.loc 
                (can be tuple of coordinates or string describing position). Defaults to 'best'.

        Returns:
            tuple: fig, ax, legend elements
        '''
        return self._plot_helper(df=self.sn_obj.reduced_df, re=re, me=me, ve=ve, of=of,
                                 figsize=figsize, errorbar=errorbar,
                                 shift=shift, ls=ls,
                                 fig=fig, ax=ax, legend=legend, legend_loc=legend_loc)

    def MCMC_trace(self, burnin=0, color=False):
        """Generates MCMC sampler chain plot.

        Args:
            burnin (int, optional): number of steps to eliminate before plotting. Defaults to 0.
            color (bool, optional): If True, each walker is a different color. Defaults to False.

        Returns:
            matplotlib.Figure: plot figure
        """
        fig, axs = plt.subplots(self.n_params, 1, figsize=(10, 16))
        for i in range(self.n_params):
            p = self.params[i]
            if color:
                axs[i].plot(np.array(self.sn_obj.MCMC_sampler[p][:, burnin:] * self.sn_obj.scale[p]).T, alpha=0.5);
            else:
                axs[i].plot(np.array(self.sn_obj.MCMC_sampler[p][:, burnin:] * self.sn_obj.scale[p]).T, color='k',
                            alpha=0.5);

            axs[i].set_title(f"{self.display_params[p]}")
        plt.suptitle(f'{self.objname} + {self.model_name} parameter sampler chains')

        fig.suptitle(f"{self.sn_obj.objname} - {self.sn_obj.display_name}")

        return fig

    def MCMC_corner(self, burnin=0):
        """Generates corner plot for the posterior distributions of parameter values.

        Args:
            burnin (int, optional): number of steps to eliminate before plotting posterior. Defaults to 0.

        Returns:
            matplotlib.Figure: plot figure
        """
        samples = (self.sn_obj.MCMC_sampler['re'][:, burnin:] * self.sn_obj.scale['re']).flatten()
        for i in range(self.n_params - 1):
            p = self.params[i+1]
            samples = np.vstack((samples, (self.sn_obj.MCMC_sampler[p][:, burnin:] * self.sn_obj.scale[p]).flatten()))

        fig = cn.corner(
            samples.T,
            labels=self.MCMC_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12, "loc": "left"})
        fig.suptitle(f"{self.sn_obj.objname} - {self.sn_obj.display_name}", x=0.7)

        
        return samples

    def make_video(self, ve=0.2, of=0.1):
        """This function will make a video depicting how the shock cooling curve changes for the given dataset
        at discrete values for mass and radius of envelope. It assumes fixed shock velocity (in 10^9 cm/s) and time offset
        (in days).

        This functionality only exists for the Sapir Waxman (BSG and RSG) models and Piro 2020.

        Args:
            ve (float, optional): shock velocity. Defaults to 0.2. (2000 km/s)
            of (float, optional): time offset from start of SN. Defaults to 0.1. (0.1 days)
        """

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
                    test_values = self.sn_obj.get_all_mags(times=phase_combo, re=r, me=m, ve=ve, of=of)
                    test_sr = self.sn_obj.get_all_mags(times=sr, re=r, me=m, ve=ve, of=of)
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