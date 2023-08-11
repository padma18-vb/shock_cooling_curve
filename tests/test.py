import argparse
from shock_cooling_curve.models import PIRO_2020
from shock_cooling_curve.models import PIRO_2015

from shock_cooling_curve import fitter

if __name__ == '__main__':
    BSG_low = [0.01, 0.01, 0.01, 0.01]
    BSG_high = [2, 2, 4, 0.5]

    RSG_low = [0.01, 0.01, 0.01, 0.01]
    RSG_high = [1, 2, 3, 0.8]

    P15_low = [0.1, 0.01, 0.001]
    P15_high = [100, 1, 1]

    P20_low = [0.1, 0.01, 0.01, 0.01]
    P20_high = [500, 1, 10, 0.2]

    CONF = 'config.ini'
    FOLD = '../data/2021gno'
    an_obj = PIRO_2020.PIRO_2020(CONF, "/Users/padmavenkatraman/Documents/Supernovae/"
                                       "SC_Modeling/SC_Notebooks/shock-cooling/data/2021gno/")
    test_fitter = fitter.Fitter(an_obj)
    test_fitter.MCMC_fit(prior_low=P20_low, prior_high=P20_high)
    # print(utils.msun)
    # print(utils.h, utils.k, utils.c)

    # test_fit = fitter.Fitter(an_obj)


    def parse_args():
        """
        Handle the command line arguments.
        Returns:
        Output of argparse.ArgumentParser.parse_args.
        """

        parser = argparse.ArgumentParser(description='Accepting config file containing supernova parameters and '
                                                     'data file.')
        parser.add_argument('-m', '--model', dest='model', type=str,
                            help='Name of model you want to apply. Options:'
                                 '1. Piro 2015\n'
                                 '2. Sapir-Waxman 2017 BSG\n'
                                 '3. Sapir-Waxman 2017 RSG\n'
                                 '4. Piro 2020\n'
                                 'Further notes: Models 2 and 3 vary in their polytropic indices (BSG; n=3, RSG; n=3/2)')

        parser.add_argument('-c', '--config', dest='config', type=str,
                            help='Config file containing the following parameters: '
                                 '1. Start time of initial shock cooling curve (MJD), \n'
                                 '2. End time of initial peak (MJD), \n'
                                 '3. Amount of dust in host galaxy, \n'
                                 '4. Amount of dust in MILKY WAY, \n'
                                 '5. Mass of core in solar masses, \n'
                                 '6. SN kinetic energy in units of 10^51 erg, \n'
                                 '7. Distance to supernova, \n'
                                 '8. Name of input .csv file.')

        parser.add_argument('-f', '--folder', dest='folder', type=str,
                            help='Folder containing you config, csv files. All your results will be stored in this folder.')
        args = parser.parse_args()
        return args


