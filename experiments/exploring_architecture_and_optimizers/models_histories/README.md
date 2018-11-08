# Models histories
In this module are the history of some relevant models is tracked. By history here we mean:
* Track final loss at each epoch
* Get train AUPRC after each epoch
* Get test AUPRC after each epoc

The relevant models are picked from the grid search involving Adam and dataset standardization:
* [The best model found](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/models_histories/MLP_scaler_adam_80_history.ipynb): the one with only 80 neurons 
* [The second best model found](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/models_histories/MLP_scaler_adam_10-2.ipynb): 2 layers with 10 and 2 neurons
* [The model that overfitted the most](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/models_histories/MLP_scaler_adam_100-80.ipynb): 2 layers with 100 and 80 neurons


As usual, there is a notebook [check_histories](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/models_histories/check_histories.ipynb) that tries to summarize the models by some plotting