This repository contains code for simulation experiments and walkthroughs on using the estimator in Alidaee, Auerbach, and Leung (2020), "Recovering Network Structure from Aggregated Relational Data Using Penalized Regression." We provide implementations in Python and R, found in their respective folders. Walkthroughs are given in the notebook files walkthrough.ipynb. The file in the Python folder can be viewed in two ways. 

1. It can be opened in your browser using this binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mpleung/ARD/master?filepath=Python%2Fwalkthrough.ipynb) Note that this might take a little bit of time to start up. 

2. You can open the file on your desktop using the notebook viewer [nteract](https://nteract.io/desktop). This method requires an installation of Python, the Python kernel, and the required modules (see instructions below), but it also allows you to use your own datasets instead of those provided in our example. To do so, save walkthrough.ipynb, nuclear\_norm\_module.py, and your datasets in CSV format to the same working directory, open the walkthrough file in nteract, and change the name of the CSVs in the walkthrough within the nteract user interface. 

The file in the R folder can also be viewed in nteract after installation of R, the R kernel, and our R package. Instructions are given below.

### Contents of Python folder ###

* walkthrough.ipynb: walkthrough of Python implementation.
* ARD\_data.csv, type\_data.csv: artificial data for the walkthrough.
* nuclear\_norm\_module.py: implementation of accelerated gradient descent method. The code builds on [this repository](https://github.com/wetneb/tracenorm).
* monte\_carlo.py: code for simulation experiments summarized in Table 2 of the paper.
* effective\_rank.py: produces Table 1 of the paper, which simulates the effective ranks of M^\* under three network formation models.

To run the .py files, we require Python 3 and installation of [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [scipy](https://www.scipy.org). See [this walkthrough](https://python.quantecon.org/getting_started.html) for installing Python. The packages can then be installed via command line by entering 

    python3 -m pip install --user numpy scipy pandas

If you are using nteract to view the walkthrough file, you also have to install an ipython kernel. See [here](https://nteract.io/kernels) for instructions. 

To execute monte\_carlo.py, download our Python folder and its contents, change your current working directory to the Python folder, and enter into command line

    python3 monte_carlo.py

### Contents of R folder ###

* walkthrough.ipynb: walkthrough of R implementation.
* ARD\_data.csv, type\_data.csv: artificial data for the walkthrough.
* nuclearARD\_0.1.tar.gz: R package for our estimator.
* nuclear\_ard: contents of R package.

You can install R [here](https://www.r-project.org/). To install our R package, download our R folder, open the R console, change the working directory to the location of our R folder, and input the following command:

    install.packages('nuclearARD_0.1.tar.gz', repos=NULL, type='source')

To view the walkthrough file in nteract, you also need to install the IRkernel. Follow the installation instructions [here](https://irkernel.github.io/installation/).
