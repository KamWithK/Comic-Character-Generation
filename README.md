# Comic Character Art Generation
This project showcases how to use common Generative Adverserial Networks for generating comic characters.
It provides an easy to use, modular and efficient framework for small GAN projects where you want to create custom networks yourself.
Variational AutoEncoders may be added in the future.

## Instalation and Environment
To simplify running code in a stable and consistant environment, please we provide an `environment.yml` file.
This file contains a list of essential packages (which certain version and channel specifications).
It is easiest to use the environment with Anaconda (or Miniconda for a smaller and more streamlined environment).


To install and use `environment.yml` do the following:
1. Install Anaconda or Miniconda
2. Create a new environemtn from the file `conda env create -f environment.yml`
3. When packages need to be added or modified update the `environment.yml` file and run `conda env update -f environment.yml --prune`
4. Once satisfied activate the new environment through `conda activate charactergen` and (optionally) deactivate/close afterwards with `conda deactivate`

*For an update the above isn't necessary - simply run `conda update --all` for packages and `conda update -n base -c defaults conda` for conda itself!*
