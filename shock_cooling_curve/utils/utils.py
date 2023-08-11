import os

import numpy as np
import pandas as pd

filter_info = pd.read_csv("/Users/padmavenkatraman/Documents/Supernovae/SC_Modeling/SC_Notebooks/"
                          "shock-cooling/shock_cooling_curve/filter_info.csv",
                          index_col=0)

rsun = 6.96e10
sigma = 5.67e-5
c = 3e10  # cm/s
msun = 1.989e+33  # g
from scipy.constants import h, k, c

h = h * 1e7
k = k * 1e7
c = c * 1e2


def get_mapping(keyA, valA, keyB):
    if keyA == 'flt':
        return filter_info.loc[valA][keyB]

    return filter_info[filter_info[keyA] == valA]


def build_offset(n):
    offset = [i for i in range(-n // 2, n // 2)]
    return offset


def get_label(flt, off):
    if off >= 0:
        return f'{flt} + {abs(off)}'
    return f'{flt} - {abs(off)}'


def BB_lam(lam, T):
    '''
        :param lam: wavelength
        :type lam: float
        :param T: temperature
        :type T: float
    '''


    lam_cm = 1e-8 * lam  # convert to cm
    rad = 2 * h * c ** 2 / (lam_cm ** 5 * (np.exp(h * c / (lam_cm * k * T)) - 1))
    radd = rad / 1e8  # erg / s / cm^2 / Ang.
    return radd


if __name__ == '__main__':
    print(os.getcwd(), os.listdir("../"))

