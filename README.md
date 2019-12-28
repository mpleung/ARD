This repository contains code for simulation experiments in Alidaee, Auerbach, and Leung (2019), "Recovering Network Structure from Aggregated Relational Data Using Penalized Regression" and an interactive example of how to use the code in practice. 

The example can be accessed at this binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mpleung/ARD/master?filepath=implementation_example.ipynb)

### Contents ###

* implementation\_example.ipynb: interactive example.
* ARD\_data.csv, type\_data.csv: artificial data for the interactive example.
* monte\_carlo.py: code for simulation experiments summarized in Table 2 of the paper.
* nuclear\_norm\_module.py: implementation of accelerated gradient descent method. The code builds on [this repository](https://github.com/wetneb/tracenorm).
* effective\_rank.py: produces Table 1 of the paper, which simulates the effective ranks of M^\* under three network formation models.

To run the .py files, we require Python 3 and installation of [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [scipy](https://www.scipy.org). After installation of Python, the packages can be installed via command line by entering 

    python -m pip install --user numpy scipy pandas


