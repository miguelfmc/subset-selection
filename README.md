# Review and Comparison of Subset Selection Methods for Linear Regression

This repository contains the relevant code for the implementation of several exact and approximate methods
for the best subset selection problem in linear regression, namely:

* Lasso
* Relaxed Lasso
* A discrete first-order method
* Best subset selection as a mixed-integer program

as well as the necessary experimental setup to run simulations of each method on data sets with varying characteristics.

This code accompanies the work and results presented in [this report](MiguelFMC_Project_SubsetSelection.pdf).

## Setup and installation

Before anything else, I advise creating a new environment, perhaps using ```conda```, and then installing the pacakge requirements.

  conda create -n <environment_name> python=3.7
  conda install --file requirements.txt
 
 In order to activate the environment just run this from the command line
 
  conda activate <environment_name>

## Real datasets

The above-mentioned work includes simulations on two real datasets: the ```prostate``` dataset and the ```lymphoma``` dataset which can be accessed from the SPLS R package.

In case you want to reproduce these experiments you will need to download these datasets and store them in the ```data```  directory.

## Running the experiments

In order to run the experiments you need to modify the ```config.py``` file to match your local setup
as well as the experiment you would like to run, encoded in the variable ```MODE```

In order to run an experiment you just need to run the ```run.py``` script from the root directory of this repository.
