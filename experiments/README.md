# Experiments
Here we have 4 folders, each of these contains the scripts for a single experiment, which explore the capabilities of the Multilayer Perceptron (MLP) in verious settings. Note that all scripts are jupyter notebooks, so it is wise to open them with an appropriate editor.

Generally speaking, in most folders under *experiments* there will be a *data* folder containing the data of the experiments. Also a *data_for_report* folder is present if necessary where all data is cleaned.

Morover there are two main kind of noetbooks:
* grid_search_*: those notebooks are those which the grid search is performed. 
* check_results_*: here the grid search results are explored, cleaned and some plots are done.


### MLP_baseline
To start, here is defined a baseline model. More information in the readme inside.
<br/>

### MLP_baseline_with_balanced_minibatch
As a refinement of the MLP_baseline, the best architecture found in MLP_baseline augmented by a balanced minibatch generator. More info inside.

### exploring_architectures_and_optimizers
This module deepen the impact of various architectures and optimizers and the need of standardization. More info inside.

### MLP_dropout_feature_decomposition_balanced_generato
Here the best architecture and optimizer found in the module exploring_architecture_and_optimizers are enhanced with various techniques that are:
* Dropout
* Regularization
* Balanced minibatch
* Feature decorrelation

Note that the terms _feature decorrelation_  _feature decomposition_, in this context, denote the process apt to reduce the number of feature in the dataset.