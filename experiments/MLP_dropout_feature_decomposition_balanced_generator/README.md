# Dropout, feature decomposition and balanced generator

In this module we will use the **ADAM** optimizer, which was found to be a good choice as the experiment in module `exploring_architectures_and_optimizers` suggests.
In folder [balanced_experiments](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments) are present the scripts regarding to balanced minibatch generator

## Grid search scripts
* [grid_search_adam_scaling_dropout_top_ten_train](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_dropout_top_ten_train.ipynb): this  grid search applies various dropout rates to the top ten models by training score -- those who overfitted the most -- found in the experiment [exploring_architectures_and_optimizers](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/exploring_architecture_and_optimizers)
* [grid_search_adam_scaling_dropout_top_ten_test](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_dropout_top_ten_test.ipynb):this  grid search applies various dropout rates to the top ten models by training score -- those who overfitted the most -- found in the experiment  [exploring_architectures_and_optimizers](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/exploring_architecture_and_optimizers)
* [grid_search_adam_scaling_decomposition_and_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_decomposition_and_dropout.ipynb):  this  grid search applies various dropout rates to the top ten models by training score -- those who overfitted the most -- found in the experiment [exploring_architectures_and_optimizers] *with a decomposed dataset*

## Dataset correlation analysis

* [dataset_correlation_analysis](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/dataset_correlation_analysis.ipynb): useful analysis on the dataset to capture correlations

## Check results

Two check reuslts files are present:
* [check_results_of_grid_search_adam_scaling_decomposition_and_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/check_results_of_grid_search_adam_scaling_decomposition_and_dropout.ipynb) this one is for the grid search with feature decomposition

* [check_results_of_grid_search_adam_scaling_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/check_results_of_grid_search_adam_scaling_dropout.ipynb) checks the results of the two grid searches without feature decomposition.