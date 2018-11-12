# Dropout, feature decomposition and balanced generator

In this module we will use the **ADAM** optimizer, which was found to be a good choice as the experiment in module `exploring_architectures_and_optimizers` suggests.
In folder [balanced_experiments](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/balanced_experiments) are present the scripts regarding the experiments with balanced minibatch generator, while folder 
[models_history](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/models_history) contains the scripts to track the history of some relevant models.
The model creation function is the following:
```python
def create_model(architecture=(100,80), dropout_rate=0.2):
    model = Sequential()
    weights_initializer = keras.initializers.glorot_normal(seed=my_seed)
    bias_init = keras.initializers.RandomNormal(mean=0.1, stddev=0.05, seed=my_seed)
    input_dim = feature_per_example
    for units in architecture:
        model.add(
            Dense(
                units,
                input_dim = input_dim,
                kernel_initializer = weights_initializer,
                bias_initializer = bias_init,
                activation="relu"
            )
        )
        model.add(
            Dropout(rate=dropout_rate, seed=my_seed)
        )
        input_dim=None # for the next layer keras infers its dimensions
        
    model.add(
        Dense(
            1,
            kernel_initializer=weights_initializer,
            bias_initializer=keras.initializers.zeros(),
            activation='sigmoid'
    ))
    optimizer = Adam()
    model.compile(loss=hingesig_tf, optimizer=optimizer)
    return model
```
## Grid search
Each grid search experiment shares the following fixed parameters for the MLP:
* weights initializer: glorot normal
* hidden layer bias init: : random normal with mean 0.1 and std 0.05
* output layer bias init:  zeros
* hidden layers activation function: ReLU
* output activation function: sigmoid
* batch size: 5000
* n_epochs: 150
* optimizer: Adam

#### [grid_search_adam_scaling_dropout_top_ten_train](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_dropout_top_ten_train.ipynb): 
this  grid search applies various dropout rates to the top ten models by training score -- those who overfitted the most -- found in the experiment [exploring_architectures_and_optimizers](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/exploring_architecture_and_optimizers). The parameter grid is 
```python
    "dropout_rate": ['0.2', '0.4, 0.6, 0.8']
    "architecture":  [(100, 80), (100, 40), (100, 80, 40), (100, 40, 20), (100, 10), (80, 40, 20), (100, 80, 50, 20), (100,), (80, 20, 10), (40, 20)]
```

#### [grid_search_adam_scaling_dropout_top_ten_test](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_dropout_top_ten_test.ipynb)
this  grid search applies various dropout rates to the top ten models by mean test AUPRC found in the experiment  [exploring_architectures_and_optimizers](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/exploring_architecture_and_optimizers). The hyperparamter grid is the following

  ```python
    "dropout_rate": ['0.2', '0.4, 0.6, 0.8']
    "architecture": [(80,), (10, 2), (100,), (40,), (20,), (100, 10), (100, 40, 20), (40, 20), (10,), (40, 10)]
  ```

#### [grid_search_adam_scaling_decomposition_and_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/grid_search_adam_scaling_decomposition_and_dropout.ipynb)
  this  grid search applies various dropout rates to the top ten models by training score -- those who overfitted the most -- found in the experiment.[exploring_architectures_and_optimizers] **with a decomposed dataset**.The hyperparameter grid is the follwing:
```python
    "dropout_rate": ['0.2', '0.4, 0.6, 0.8']
    "architecture":  [(100, 80), (100, 40), (100, 80, 40), (100, 40, 20), (100, 10), (80, 40, 20), (100, 80, 50, 20), (100,), (80, 20, 10), (40, 20)]
```

## Dataset correlation analysis

 [dataset_correlation_analysis](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/dataset_correlation_analysis.ipynb): useful analysis on the dataset to capture correlations

## Check results

Two check reuslts files are present:
#### [check_results_of_grid_search_adam_scaling_decomposition_and_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/check_results_of_grid_search_adam_scaling_decomposition_and_dropout.ipynb) 
this one is for the grid search with feature decomposition

#### [check_results_of_grid_search_adam_scaling_dropout](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_dropout_feature_decomposition_balanced_generator/check_results_of_grid_search_adam_scaling_dropout.ipynb) 
checks the results of the two grid searches without feature decomposition.
