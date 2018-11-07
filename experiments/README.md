# Experiments
Here we have 4 folders, each of these contains the scripts for a single experiment, which explore the capabilities of the Multilayer Perceptron (MLP) in verious settings. Note that all scripts are jupyter notebooks, so it is wise to open them with an appropriate editor.
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