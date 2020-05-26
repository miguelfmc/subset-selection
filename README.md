# Review and Comparison of Subset Selection Methods for Linear Regression

This repository contains the relevant code for the implementation of several exact and approximate methods
for the best subset selection problem in linear regression, namely:

* Lasso
* Relaxed Lasso
* A discrete first-order method
* Best subset selection as a mixed-integer program

as well as the necessary experimental setup to run simulations of each method on data sets, both synthetic and real, with varying characteristics.

This code accompanies the work and results presented in [this report](MiguelFMC_Project_SubsetSelection.pdf).

The main goal of the project was to compare the performance of the methods mentioned above across a variety of evaluation metrics for statistical performance and sparsity.

## Setup and installation

Before anything else, I advise creating a new environment, perhaps using ```conda```, and then installing the package requirements.

One option is to use the ```environment.yml``` file:

```
conda env create -f environment.yml
```

Another option is to create a environment and then install the requirements using the ```requirements.txt``` file.

```
conda create -n <environment_name> python=3.7
conda activate <environment_name>
conda install --file requirements.txt
```

Finally, you can also use the package manager ```pip```
 
In order to activate the environment just run this command from the command line:

```
conda activate <environment_name>
```

### Gurobi

Note that this work uses the Gurobi Python interface to solve a mixed-integer program.
In order to use Gurobi you need a license.
More information can be found [here](https://www.gurobi.com/).

## Real datasets

The above-mentioned work includes simulations on two real datasets: the ```prostate``` dataset and the ```lymphoma``` dataset which can be accessed from the [SPLS R package](https://cran.r-project.org/web/packages/spls/index.html).

In case you want to reproduce these experiments you will need to download these datasets and store them in the ```data```  directory.

## Running the experiments

In order to run the experiments you need to modify the ```config.py``` file to match your local setup
as well as the experiment you would like to run, encoded in the variable ```MODE```.

In order to run an experiment you just need to run the ```run.py``` script from the root directory of this repository, for example:

```
$ python src/run.py
```
