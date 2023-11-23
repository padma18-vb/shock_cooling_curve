import numpy as np
import pandas as pd
import importlib.resources as pkg_resources


path = pkg_resources.path('shock_cooling_curve', 'filter_info.csv')
with path as p:
    filter_info = pd.read_csv(p, index_col=0)

rsun = 6.96e10
sigma = 5.67e-5
c = 3e10  # cm/s
msun = 1.989e+33  # g
from scipy.constants import h, k, c
# convert to CGS units
h = h * 1e7
k = k * 1e7
c = c * 1e2

def get_absolute_mag(app_mag, dist, kcorr=0):
    """Provided an apparent magnitude and distance in parsec, compute absolute magnitude.
    No 

    Args:
        app_mag (float): apparent magnitude
        dist (float): Distance of SN in pc
        kcorr (float, optional): K-correction to bring to restframe redshift. Defaults to 0.

    Returns:
        float: absolute magnitude value
    """
    return app_mag - 5*np.log10(dist/10) - kcorr


def get_mapping(keyA, valA, keyB):
    """Obtains mapping between two columns in the filter table

    Args:
        keyA (str): first column
        valA (str or float): value in first column
        keyB (str): second column to find match in

    Returns:
        str or float: mapped value in column keyB.
    """
    if keyA == 'flt':
        return filter_info.loc[valA][keyB]

    return filter_info[filter_info[keyA] == valA]


def build_offset(n):
    """Builds an equally spaced offset list of length n centered on 0.

    Args:
        n (int): number of elements

    Returns:
        list: list of integers containing n elements centered at 0.
    """
    offset = [i for i in range(-n // 2, n // 2)]
    return offset


def get_label(flt, off):
    """Gets plotting legend label given a filter and the magnitude shift applied to it

    Args:
        flt (str): filter
        off (int): offset

    Returns:
        str: label describing filter
    """
    if off > 0:
        return f'{flt} + {abs(off)}'
    elif off == 0:
        return f'{flt}'
    return f'{flt} - {abs(off)}'


def BB_lam(lam, T):
    """Planck blackbody function at a given wavelength and temperature

    Args:
        lam (float): wavelength in angstroms (10^10 m)
        T (temperature): temperature in Kelvin

    Returns:
        float: blackbody flux
    """
    lam_cm = 1e-8 * lam  # convert to cm
    rad = 2 * h * c ** 2 / (lam_cm ** 5 * (np.exp(h * c / (lam_cm * k * T)) - 1))
    radd = rad / 1e8  # erg / s / cm^2 / Ang.
    return radd

