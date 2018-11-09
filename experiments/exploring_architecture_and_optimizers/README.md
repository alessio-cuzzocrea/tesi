# Exploring architectures and optimizers

The purpose of this experiment is to explore a big variety of MLP architectures and to evaluate a how a different optimizer affects the performance. Two  two different optimizers are evaluated:
* Adam
* SGD



Along with a carefull choice of the minibatch size: 5000 sample per batch.
<br/>
And this is how the model is created. Of course the optimizer should be changed accordingly.

```python
def create_model(architecture=(10,)):
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
        Dense(
            1,
            kernel_initializer=weights_initializer,
            bias_initializer=keras.initializers.zeros(),
            activation='sigmoid'
    ))
    optimizer = SGD()
    model.compile(loss=hingesig_tf, optimizer=optimizer)
    return model
model = KerasClassifier(build_fn=create_model, verbose=1, shuffle=True, batch_size=batch_size, epochs=150)
```

## Grid search
The fixed parameters are:
* weights initializer: glorot normal
* hidden layer bias init: : random normal with mean 0.1 and std 0.05
* output layer bias init:  zeros
* hidden layers activation function: ReLU
* output activation function: sigmoid
* batch size: 5000
* n_epochs: 150

Each grid search experiment have the following paramter grid:

```
"architecture": [(2), (5), (10), (20), (40), (80),(100),(100, 80), (100, 40), (100, 10), (40, 20), (40,10), (20,10), (20,5), (10, 5), (10,2), (100, 80, 40), (100,40,20),(80,40,20), (80, 20,10), (40,20,10),(20,10,5), (10,5,2), (100,80,50,20), (100,50,25,10), (80, 60, 20,10), (50, 30, 20, 10),(30,15,7,3) ]
```
#### [grid_search_MLP_architectures_no_scaler_SGD](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/grid_search_MLP_architectures_no_scaler_SGD.ipynb)
This is a grid search using SGD and w/o standardization of the dataset.

#### [grid_search_MLP_architectures_no_scaler_adam](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/grid_search_MLP_architectures_no_scaler_adam.ipynb)
This is a grid search using Adam optimizer and no standardization

#### [grid_search_MLP_architectures_no_scaler_adam.ipynb](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/grid_search_MLP_architectures_no_scaler_adam.ipynb)
This is a grid sarch involving adam w/ standardization
#### [grid_search_MLP_architectures_scaler_SGD](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/grid_search_MLP_architectures_scaler_SGD.ipynb)
This is a grid search using SGD w/ standardization

#### [grid_search_MLP_architectures_scaler_adam_gpu_run](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/grid_search_MLP_architectures_scaler_adam_gpu_run.ipynb)
This is the  same grid search of the one with Adam and scaling which purpose is to evaluated the impact of a GPU in terms of computation.


## Results Check
Other files, useful for model presentation, visualization and assessment.
#### [check_results_of_grid_search_MLP_architecture](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/check_results_of_grid_search_MLP_architecture.ipynb)
Here all the grid search results are visualized by plotting some relevant informations.

#### [check_results_of_grid_search_MLP_architectures_scaler_adam_gpu](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/check_results_of_grid_search_MLP_architectures_scaler_adam_gpu.ipynb)
Check the results of the GPU run.

## Other files and folders
#### [minibatch_positive_sample_size_probability](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/exploring_architecture_and_optimizers/minibatch_positive_sample_size_probability.ipynb)
The purpose of this script is to highlight the importance of choicing carefully the batch size

#### [unusual_aurpc_curve/](https://github.com/alessio-cuzzocrea/tesi/tree/master/experiments/exploring_architecture_and_optimizers/unusual_aurpc_curve)
In this folder is shown an extreme case where the sklearn `precision_recall_curve` calculaction could drive to an erroneous AUC, but there is `average_precision_score` that summarizes the PR-curve accurately