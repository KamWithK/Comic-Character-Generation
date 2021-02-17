# The Lionel Polanski Project: Comic Character Art Generation
## Why Lionel Polanski?
Lionel Polanski is such an enigma (and nonetheless, a very cool name)

## Instalation and Environment
To simplify running code in a stable and consistant environement, please we provide an `environment.yaml` file.
This file contains a list of essential packages (which certain version and channel specifications).
It is easiest to use the environment with Anaconda (or Miniconda for a smaller and more streamlined environment).


To install and use `environment.yaml` do the following:
1. Install Anaconda or Miniconda
2. Create a new environemtn from the file `conda env create -f environment.yaml`
3. When packages need to be added or modified update the `environment.yaml` file and run `conda env update -f environment.yaml --prune`
4. Once satisfied activate the new environment through `conda activate ct` and (optionally) deactivate/close afterwards with `conda deactivate`

*For an update the above isn't necessary - simply run `conda update --all` for packages and `conda update -n base -c defaults conda` for conda itself!*
