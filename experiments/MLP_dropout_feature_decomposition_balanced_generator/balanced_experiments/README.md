# Balanced experiments
In this folder we apply the balanced minibatch generator to the best model found in the parent folder, which is the one with the following paramenters:
* Architecture: (100, )
* Dropout rate: 0.2

## Grid search experiments

* [grid_search_adam_scaling_dropout_mb_gen](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen.ipynb): here we setup a grid search to explore various settings of the balanced minibatch generator. The hyperparameters in the search space are:     
    * negative_perc: [1],
    * positive_sample_perc: [0.5, 1, 1.5],
    * np_ratio: [1, 3,5, 10, 15] <br/>


* [grid_search_adam_scaling_dropout_mb_gen_l2_norm](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen_l2_norm.ipynb): here we setup a grid search to explore the capabilities of the L2 norm applied to the balanced minibatch generator. The hyper parameter search space is:
    * alpha: [1e-15, 1e-1,1e-5,1]

* [grid_search_adam_scaling_dropout_mb_gen_maxnorm](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/grid_search_adam_scaling_dropout_mb_gen_maxnorm.ipynb): here we setup a gridsearch to get an insight on how the maxnorm behaves with the minibatch balanced generator. The hyperparameters are:
    * c: [3,3.5,4]

## Check results and plotting

In notebook  [balanced_experiments/check_results_of_mb_gen](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments/check_results_of_mb_gen.ipynb) the results are summarized