# Models history

In this folderthe best model found in each grid search are trained over the whole training set and their performace is tracked. Within each script are some relevant plots like PR-ROC curve.

* [adam_scaling_100_units_dropout-0.2_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history/adam_scaling_100_units_dropout-0.2_history.ipynb): this the history for the best model found in [grid_search_adam_scaling_dropout_top_ten_train](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_dropout_top_ten_train.ipynb) which has the following parameters:
    * dropout rate: 0.2
    * architecture: one hidden layer, 100 units

* [adam_scaling_100-units_dropout-0.2_decomposition_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history/adam_scaling_100-units_dropout-0.2_decomposition_history.ipynb): this is the history of  the best model found in the experiment [grid_search_adam_scaling_decomposition_and_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_decomposition_and_dropout.ipynb) with the following features\parameters
    * dropout rate: 0.2
    * architecture: one hiddenlayer, 100 units
    * dataset decomposition

* [adam_scaling_100-units_dropout-0.2_mb_gen_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history/adam_scaling_100-units_dropout-0.2_mb_gen_history.ipynb) this is the history of the best model found in [grid_search_adam_scaling_dropout_mb_gen](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen.ipynb) with the following paramters:
  * dropout rate: 0.2
  * architecture: one hidden layer, 100 units
  * np_ratio: 3
  * positive_sample_perc: 1.5

* [adam_scaling_100-units_dropout-0.2_mb_gen__l2_history.ipynb](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history/adam_scaling_100-units_dropout-0.2_mb_gen__l2_history.ipynb): this is history of the best model found in [grid_search_adam_scaling_dropout_mb_gen_l2_norm](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen_l2_norm.ipynb)
  * dropout rate: 0.2
  * architecture: one hidden layer, 100 units
  * np_ratio: 3
  * positive_sample_perc: 1.5
  * weights regularization: L2 norm

* [adam_scaling_100-units_dropout-0.2_mb_gen_maxnorm_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history/adam_scaling_100-units_dropout-0.2_mb_gen_maxnorm_history.ipynb) this is the history of the best model found in [balanced_experiments/grid_search_adam_scaling_dropout_mb_gen_maxnorm](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen_maxnorm.ipynb)
  * dropout rate: 0.2
  * architecture: one hidden layer, 100 units
  * np_ratio: 3
  * positive_sample_perc: 1.5
  * weights regularization: maxnorm
