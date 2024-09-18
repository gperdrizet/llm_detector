'''Collection of functions for XGBoost classifier training/evaluation with
length binned data. Parallelizes models over the bins.'''
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from joblib import parallel_backend

import functions.notebook_helper as helper_funcs

def cross_validate_bins(
        input_file: str,
        scoring_funcs: dict,
        parsed_results: dict,
        num_workers: int,
        shuffle_control: bool = False
) -> dict:

    '''Main function to parallelize cross-validation of XGBoost classifier
    over the data's length bins.'''

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(input_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Instantiate worker pool
    pool = mp.Pool(
        processes = num_workers,
        maxtasksperchild = 1
    )

    # Holder for returns from workers
    async_results = []

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Loop on the bins
    for worker_num, bin_id in enumerate(bins.keys()):

        # Pull the training features for this bin
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Pull the training labels for this bin
        bin_training_labels = data_lake[f'training/{bin_id}/labels']

        async_results.append(
            pool.apply_async(cross_validate_bin,
                args = (
                    bin_training_features_df, 
                    bin_training_labels,
                    scoring_funcs,
                    worker_num,
                    bin_id,
                    shuffle_control
                )
            )
        )

    # Clean up
    pool.close()
    pool.join()

    # Get the results
    results = [async_result.get() for async_result in async_results]

    print()

    # Add the results
    for result in results:

        bin_id = result[0]
        scores = result[1]
        parsed_results = helper_funcs.add_cv_scores(parsed_results, scores, bin_id)

    data_lake.close()

    return parsed_results


def cross_validate_bin(
        features_df: pd.DataFrame, 
        labels: pd.Series,
        scoring_funcs: dict,
        worker_num: int, 
        bin_id: str,
        shuffle_control: bool
) -> tuple[str, dict]:
    
    '''Runs cross-validation on bin data with XGBClassifier'''

    # Clean up the features for training
    features = prep_data(features_df, shuffle_control)

    # Do the cross-validation training run
    results = run_cross_validation(features, labels, scoring_funcs)

    return bin_id, results


def prep_data(features_df: pd.DataFrame, shuffle_control: bool) -> pd.DataFrame:
    '''Takes a dataframe of features, drops unnecessary or un-trainable features,
    cleans up missing data, shuffles and returns updated dataframe.'''

    # Next, get rid of un-trainable/unnecessary features
    feature_drops = [
        'Fragment length (words)',
        'Source',
        'String'
    ]

    features_df.drop(feature_drops, axis = 1, inplace = True)

    # Replace and remove string 'OOM' and 'NAN' values
    features_df.replace('NAN', np.nan, inplace = True)
    features_df.replace('OOM', np.nan, inplace = True)
    features_df.dropna(inplace = True)

    # Last, if this is a shuffle control run, randomize the order of the
    # data. This will break the correspondence between features and labels
    if shuffle_control == True: 
        features_df = features_df.sample(frac = 1).reset_index(drop = True)

    return features_df


def run_cross_validation(features: pd.DataFrame, labels: pd.Series, scoring_funcs: dict) -> dict:
    '''Does the cross-validation'''

    # Use threading backend for parallelism because we are running in a
    # daemonic worker process started by multiprocessing and thus can't 
    # use multiprocessing again to spawn more processes
    with parallel_backend('threading', n_jobs = 2):

        # Instantiate sklearn gradient boosting classifier
        model = XGBClassifier(random_state = 42)

        # Run cross-validation
        scores = cross_validate(
            model,
            features.to_numpy(),
            labels.to_numpy(),
            cv = 3,
            n_jobs = 2,
            scoring = scoring_funcs
        )

    return scores