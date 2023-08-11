# shock-cooling-curve
## INTRODUCTION
This package can be used to fit double-peaked supernova light curves using the following shock cooling emission models:

1. [PIRO 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...808L..51P/abstract)
2. [Sapir-Waxman 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...838..130S/abstract)
   
   A) For a blue supergiant (n = 3)
   
   B) For a red supergiant (n = 3/2)
   
3. [PIRO 2020](https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract)

## Using this package
There are two files that have to be prepared before using this package.
1. The config file: A template for this file in provided under templates/config_template.ini. Simply make a copy of the
file and fill out all the entries in the DEFAULT section. The BOUNDS section is optional.
   
2. The data file: This file containing the photometry data must be a csv. The column headers and template 
   is under templates/phot_template.csv.
   
Related Papers:
[The Circumstellar Environments of Double-Peaked, Calcium-strong Supernovae 2021gno and 2021inl](https://arxiv.org/abs/2203.03785)
