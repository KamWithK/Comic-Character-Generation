# The Lionel Polanski Project: Comic Character Art Generation
## Why Lionel Polanski?
Lionel Polanski is such an enigma (and nonetheless, a very cool name)

## Instalation and Environment
To simplify running code in a stable and consistant environment, please we provide an `environment.yaml` file.
This file contains a list of essential packages (which certain version and channel specifications).
It is easiest to use the environment with Anaconda (or Miniconda for a smaller and more streamlined environment).


To install and use `environment.yaml` do the following:
1. Install Anaconda or Miniconda
2. Create a new environemtn from the file `conda env create -f environment.yaml`
3. When packages need to be added or modified update the `environment.yaml` file and run `conda env update -f environment.yaml --prune`
4. Once satisfied activate the new environment through `conda activate ct` and (optionally) deactivate/close afterwards with `conda deactivate`

*For an update the above isn't necessary - simply run `conda update --all` for packages and `conda update -n base -c defaults conda` for conda itself!*

## Developer Instructions
### Custom Configuration Options
To change the way you run your experiments you can add in custom arguments in the `config.json` file.
These are fed straight into their relavent modules:
* `data_args` into the data module
* `model_args` into the model
* `trainer_args` into the trainer

For extra flexability there are custom modes which allow you to define special sets of options which you can switch between.
Set the mode argument to any string (by default "debug") and the code will look for a matching mode set within `config.json` (like "debug_mode").

### Tracking Experiments with Weights and Biases
To log your experiments to Weights and Biases, log into your account through the command `wandb login` using an [API key](https://app.wandb.ai/authorize).
Once you've done this you're ready to run your code.
Make sure to fill in all the information on the experiment (within the `config.json` file's `experiment_args` criteria):
* `name` - the name of the experiment (short and sweet)
* `id` - a unique string to represent the run (for resumability)
* `tags` - a list (like `["first tag", "second tag", "third tag"]`) of tags (stay consistant, so filtering and visualising works)
* `group` - a string to group multiple runs together
* `log_model` - whether to upload the model itself (can be slow post-run, so avoid whilst debugging)

**To resume a previous experiment, make sure to specify the `id` and model checkpoints path**!
You can find a previous run's id by looking at its url - *https://app.wandb.ai/kamwithk/ct-facial-reconstruction/runs/***ID**.
Remember that you can always add/modify any properties on the [Weights and Biases Dashboard](https://app.wandb.ai/monash-deep-neuron/ct-facial-reconstruction) during and after a run.

### Debugging
We currently are moving from pure PyTorch to PyTorch lightning code.
This gives us more flexibility, and provides several neat ways to debug a model to ensure everything's working before starting training.

Here's a brief summary of useful arguments to pass into the trainer to ensure the code/model works:
* `fast_dev_run` - run a single training and validation batch
* `overfit_batches` - test ability for model to overfit some percent of training data
* `limit_train_batches` and `limit_val_batches` - train and test with some percentage or some number of batches
* `train_percent_check` and `val_percent_check` - train and test on some percent of data
* `num_sanity_val_steps` - run some number of validation loops to test for failures (default 2)

### Resuming Run
The `resume_from_checkpoint` argument specifies the path to the checkpoint you want to load (to continue training from a presaved state) and ensure to provide a WandB `id` (to resume a run).

Please refer to the [PyTorch Lightning debugging page](https://pytorch-lightning.readthedocs.io/en/latest/debugging.html).
There are several other useful arguments for [debugging](https://pytorch-lightning.readthedocs.io/en/latest/debugging.html) and other purposes (refer to the [official docs](https://pytorch-lightning.readthedocs.io/) and [API](https://pytorch-lightning.readthedocs.io/en/latest/api))!
These can save time and prevent wasting time training a model which couldn't work/code that crashes afterwards.

*Not that the arguments/features of PyTorch Lightning only apply to the newer in progress training scripts and notebooks.*
