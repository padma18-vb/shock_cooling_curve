# shock-cooling-curve
## Introduction
This package can be used to fit double-peaked supernova light curves using the following shock cooling emission models:

1. [PIRO 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...808L..51P/abstract)
2. [Sapir-Waxman 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...838..130S/abstract)
   
   A) For a blue supergiant (n = 3)
   
   B) For a red supergiant (n = 3/2)
   
3. [PIRO 2020](https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract)

## Package Setup
### Before installation
This package utilizes `pysynphot` in order to create synthetic photometry generated according to the analytical models 
used. The user need to install `pysynphot` before installing `shock_cooling_curve` which includes downloading required
`pysynphot` files and adding them to user path.

#### `pysynphot` instructions:
Detailed instructions for how to set up `pysynphot` on your system are provided [here](https://pysynphot.readthedocs.io/en/latest/index.html#installation-and-setup).
Here is the truncated version adapted from `pysynphot` provided guidelines:
1. `pip install pysynphot`
2. Two sets of `tar` files, [1](http://ssb.stsci.edu/trds/tarfiles/synphot1.tar.gz) and [2](http://ssb.stsci.edu/trds/tarfiles/synphot2.tar.gz)
need to be downloaded and store in some local directory.
3. The terminal source file (accessible by calling `vi .zprofile` on MAC) should be opened and edited to include the
path to the `pysynphot` files set by `export PYSYN_CDBS=/my/local/dir/trds/`. Note that the variable **should** be
`PYSYN_CDBS`.
4. Check if this is done correctly by opening python in your terminal and calling the following command:
```commandline
import os
os.environ['PYSYN_CDBS']
>>> '/my/local/dir/trds/'
```
Once you have this setup, you should be good to install `shock_cooling_curve`!

### Contributing
Any code changes, suggestions or improvements are welcome and can be submitted by making a PR! To develop this code, you
can:
1. Fork this repository. It will appear in your own GitHub account as https://github.com/<your_username>/shock_cooling_curve.
2. Clone your forked `shock_cooling_curve` repository
3. `cd` into the folder `shock_cooling_curve` and `pip install -e . `

## Using this package
There are two files that have to be prepared before using this package.
1. The config file: A template for this file in provided under templates/config_template.ini. Simply make a copy of the
file and fill out all the entries in the DEFAULT section. The BOUNDS section is optional.
   
2. The data file: This file containing the photometry data must be a csv. The column headers and template 
   is under templates/phot_template.csv.

3. If you are unsure about the naming convention of filters when you include them in your photometry file, you can refer
to [filter_info.csv](https://github.com/padma18-vb/shock_cooling_curve/blob/main/templates/filter_info.csv) under templates.
   
Related Papers:
[The Circumstellar Environments of Double-Peaked, Calcium-strong Supernovae 2021gno and 2021inl](https://arxiv.org/abs/2203.03785)
