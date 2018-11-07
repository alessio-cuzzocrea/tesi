# Balanced minibatch generator

Here we explore how a balanced minibatch generator affect the best model found in [MLP_baseline](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/MLP_baseline), which is the one with hingeloss and sigmoid activation function. The model is defined as


    def create_model():
        model = Sequential()
        initializer = keras.initializers.glorot_uniform(seed=my_seed)
        model.add(Dense(
                300, 
                input_dim=feature_per_example, 
                kernel_initializer=initializer,
                activation="sigmoid")
                )
        model.add(Dense(
                1,
                kernel_initializer=initializer,
                activation='sigmoid'
        ))
        optimizer = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
        model.compile(loss=hingesig_tf, optimizer=optimizer)
        return model
    model = CustomKerasClassifier(build_fn = create_model, generator=gen, verbose=1, shuffle=False)

### Experiments
* [grid_search_MLP_baseline_with_minibatch_balanced_generator](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/grid_search_MLP_baseline_with_minibatch_balanced_generator.ipynb) here we exhaustively try all hyperparamters defined in the hyperparameter grid.

* [grid_search_MLP_baseline_with_minibatch_balanced_generator_seed_reset](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/grid_search_MLP_baseline_with_minibatch_balanced_generator_seed_reset.ipynb) This experiment is the same of the previous but the seed is reset when a new model begins the cross validation. This experiment has been run because the previous one it's not reproducible, and neither this.

### check_results
In these files we analyze the grid search results
* [check_results_of_grid_search_MLP_baseline_with_minibatch_balanced_generator](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/check_results_of_grid_search_MLP_baseline_with_minibatch_balanced_generator.ipynb)

* [check_results_of_grid_search_MLP_baseline_with_minibatch_balanced_generator_seed_reset](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/check_results_of_grid_search_MLP_baseline_with_minibatch_balanced_generator_seed_reset.ipynb)

### MLP_baseline_balanced_mb_300_epochs_runs

The best five models and the the most overfitting five found in the grid search are evaluate on a much longer run with some plotting.
* [top_5_test_MLP_baseline_mb_gen_300_epochs_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/top_5_test_MLP_baseline_mb_gen_300_epochs_history.ipynb)

* [top_5_train_MLP_baseline_mb_gen_300_epochs_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/top_5_train_MLP_baseline_mb_gen_300_epochs_history.ipynb)
* [plots_for_MLP_baseline_mb_gen_with_300_epochs](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/plots_for_MLP_baseline_mb_gen_with_300_epochs.ipynb)