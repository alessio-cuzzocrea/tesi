# 300 epochs run
The MLP fixed parameters are the same of the one specified in the parent folder.

### [top_5_test_MLP_baseline_mb_gen_300_epochs_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/top_5_test_MLP_baseline_mb_gen_300_epochs_history.ipynb)
Here we evalute in a run with 300 epochs the performance of the best five models found in the parent experiment. Those models are:

| negative_perc| np_ratio| positive_sample_perc|
|--------------|---------|---------------------|
| 0.25 | 0.5 | 1 |
| 0.5 | 0.5 | 1 | 
|0.25 | 3 | 1 |
| 0.25| 1 | 0.5 |
|0.5| 1.5 | 0.25
### [top_5_train_MLP_baseline_mb_gen_300_epochs_history](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/top_5_train_MLP_baseline_mb_gen_300_epochs_history.ipynb)

| negative_perc| np_ratio| positive_sample_perc|
|--------------|---------|---------------------|
| 1| 5 | 0.25 |
| 1| 3 | 0.25 | 
| 1| 3 | 0.75 |
| 1| 3 | 0.5 |
| 1| 1 | 0.75

### [plots_for_MLP_baseline_mb_gen_with_300_epochs](https://github.com/alessio-cuzzocrea/tesi/blob/master/experiments/MLP_baseline_with_balanced_minibatch/MLP_baseline_balanced_mb_300_epochs_runs/plots_for_MLP_baseline_mb_gen_with_300_epochs.ipynb)
In this notebook there are some relevant plots, like AUPRC history over epochs.