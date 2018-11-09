import pandas as pd
import numpy as np

def get_mendelian_dataset(return_df=False):
    """reads the mendelian dataset used for the master thesis of Alessio Cuzzocrea. 
        Important note: this function is not general purpose, it's intended to be used in a specific environment.
    
    Keyword Arguments:
        return_df {bool} -- wheter to return the mendelian dataset as a pandas DataFrame or a simple numpy array
    
    Returns:
        A 4-uple -- train_X, train_y, test_X, test_y
    """
    
    base_data_folder = "/home/alessio/dati/"
    train_set_filename = base_data_folder + "Mendelian.train.tsv"
    test_set_filename = base_data_folder + "Mendelian.test.tsv"

    train_X = pd.read_csv(train_set_filename, sep='\t')
    test_X = pd.read_csv(test_set_filename, sep='\t')
    #creiamo le label, nel train set i primi 356 esempi sono positivi, nel test i primi 40 sono positivi
    n_positives = 356
    n_negatives = train_X.shape[0] - n_positives
    train_y = np.concatenate((
        np.ones(n_positives, dtype=np.int32),
        np.zeros(n_negatives, dtype=np.int32)
    ))
    n_positives = 40
    n_negatives = test_X.shape[0] - n_positives
    test_y = np.concatenate((
        np.ones(n_positives, dtype=np.int32),
        np.zeros(n_negatives, dtype=np.int32)
    ))
    if return_df:
        return train_X, train_y, test_X, test_y
    else:
        return train_X.values, train_y, test_X.values, test_y
