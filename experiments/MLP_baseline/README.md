# MLP baseline

Basically here we want to have a first glance to neural networks by using a very basic model, the list of dependencies is defined in `requirements.txt`. The model is defined by the following function:



 ```python
def create_model(loss='binary_crossentropy', activation='tanh'):
    model = Sequential()
    initializer = keras.initializers.glorot_uniform(seed=my_seed)
    model.add(Dense(
            300, 
            input_dim=feature_per_example, 
            kernel_initializer=initializer,
            activation=activation)
            )
    model.add(Dense(
            1,
            kernel_initializer=initializer,
            activation='sigmoid'
    ))
    optimizer = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
    model.compile(loss=loss, optimizer=optimizer)
    return model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size=batch_size, verbose=1, shuffle=False)
```
## Grid search
In this experiment the fixed MLP parameters are:
* hidden layers: 1
* hidden units: 300
* optimizer: SGD
* batch size: 500
* n_epochs: 10
* weights initializer: glorot uniform
* output activation function: sigmoid
#### [grid_search_MLP_loss_and_activations]( https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline/grid_search_MLP_loss_and_activations.ipynb)
In this file there is the exhaustive search of each combination of the following hyperparameters:
```python
params = {
        'loss': ['binary_crossentropy', 'mean_absolute_error', hingesig],
        'activation': ['softmax', 'softplus', 'tanh', 'relu', 'sigmoid']
    }
```
## Single model evaluation
#### [MLP_hingeloss_sigmoid](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline/MLP_hingeloss_sigmoid.ipynb)
Here we evaluate the MLP with theparameters  ```loss=hingesig``` and ```activation=sigmoid``` and do some basic plots.
####  [MLP_hingeloss_sigmoid_with_shuffling](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline/MLP_hingeloss_sigmoid_with_shuffling.ipynb) 
Here we evaluate the MLP with theparameters  ```loss=hingesig``` and ```activation=sigmoid``` and do some basic plots, but the the training data  is shuffled.
## Other files
There is also a check_auprc.R file that computes the AUPRC with R, and two csv files containing the grid search results and the results cleaned for reporting.
