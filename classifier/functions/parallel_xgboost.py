'''Collection of functions for XGBoost classifier training/evaluation with
length binned data. Parallelizes models over the bins.'''

from __future__ import annotations

import os
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from joblib import parallel_backend

import functions.notebook_helper as helper_funcs

def cross_validate_bins(
        input_file: str,
        scoring_funcs: dict,
        parsed_results: dict,
        cv_folds: int = 5,
        workers: int = 3,
        shuffle_control: bool = False
) -> dict:

    '''Main function to parallelize cross-validation of XGBoost classifier
    over the data's length bins.'''

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(input_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Batch the bins into chunks of size worker number
    bin_ids = list(bins.keys())
    bin_batches = [bin_ids[i:i + workers] for i in range(0, len(bin_ids), workers)]

    # Batch CPUs so we can pin workers to them with taskset linux command.
    # Note: pretty sure it's something about scikit-learn that makes this 
    # necessary, or the interaction of scikit-learn's use of multiprocessing, 
    # combined with our use of multiprocessing and/or joblib threading on 
    # top of all of that. If we don't do it this way, every worker tries to 
    # use all of the CPUs.
    CPU_batches = batch_cpus(workers)

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Loop on the bins
    for bin_batch in bin_batches:

        # Instantiate worker pool.
        pool = mp.Pool(
            processes = workers,
            maxtasksperchild = 1
        )

        # Holder for worker returns
        async_results = []

        for bin_id, bin_CPUs in zip(bin_batch, CPU_batches):

            print(f'Running cross-validation on {bin_id} with {len(bin_CPUs)} threads, CPUs: {", ".join(bin_CPUs)}')

            # Pull the training features for this bin
            bin_training_features_df = data_lake[f'training/{bin_id}/features']

            # Pull the training labels for this bin
            bin_training_labels = data_lake[f'training/{bin_id}/labels']

            async_results.append(
                pool.apply_async(cross_validate_bin,
                    args = (
                        bin_training_features_df, 
                        bin_training_labels,
                        cv_folds,
                        scoring_funcs,
                        bin_id,
                        bin_CPUs,
                        shuffle_control
                    )
                )
            )

        # Clean up
        pool.close()
        pool.join()

        # Get the results
        results = [async_result.get() for async_result in async_results]

        # Add the results
        for result in results:

            bin_id = result[0]
            scores = result[1]

            parsed_results = helper_funcs.add_two_factor_cv_scores(
                parsed_results,
                scores,
                bin_id,
                False
            )

    data_lake.close()

    return parsed_results


def cross_validate_bin(
        features_df: pd.DataFrame, 
        labels: pd.Series,
        cv_folds: int,
        scoring_funcs: dict,
        bin_id: str,
        bin_CPUs: list[str],
        shuffle_control: bool
) -> tuple[str, dict]:
    
    '''Runs cross-validation on bin data with XGBClassifier'''

    # Clean up the features for training
    features = prep_data(features_df, shuffle_control)

    # Do the cross-validation training run
    results = run_cross_validation(
        features = features, 
        labels = labels, 
        cv_folds = cv_folds, 
        scoring_funcs = scoring_funcs,
        bin_CPUs = bin_CPUs
    )

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


def run_cross_validation(
        features: pd.DataFrame, 
        labels: pd.Series, 
        cv_folds: int, 
        scoring_funcs: dict,
        bin_CPUs: list[str]
) -> dict:
    
    '''Does the cross-validation'''

    # Assign CPUs to this process. Note: pretty sure it's something
    # about scikit-learn that makes this necessary, or the interaction
    # of scikit-learn's use of multiprocessing, combined with our use
    # of multiprocessing and/or joblib threading on top of all of that.
    # If we don't do it this way, every worker tries to use all of the
    # CPUs.
    pid = os.getpid()
    cmd = 'taskset -cp %s %i' % (','.join(bin_CPUs), pid)
    _ = os.popen(cmd).read()

    # Use threading backend for parallelism because we are running in a
    # daemonic worker process started by multiprocessing and thus can't 
    # use multiprocessing again to spawn more processes
    with parallel_backend('threading', n_jobs = len(bin_CPUs)):

        # Instantiate sklearn gradient boosting classifier
        model = XGBClassifier(random_state = 42)

        # Run cross-validation
        scores = cross_validate(
            model,
            features.to_numpy(),
            labels.to_numpy(),
            cv = cv_folds,
            n_jobs = len(bin_CPUs),
            scoring = scoring_funcs
        )

    return scores

def hyperparameter_tune_bins(
        input_file: str,
        parameter_distributions: dict,
        scoring_funcs: dict,
        cv_folds: int = 5,
        n_iterations: int = 3, 
        workers: int = 3, 
        shuffle_control: bool = False
) -> dict:

    '''Main function to parallelize hyperparameter optimization of XGBoost classifier
    over the data's length bins.'''

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(input_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Batch the bins into chunks of size worker number
    bin_ids = list(bins.keys())
    bin_batches = [bin_ids[i:i + workers] for i in range(0, len(bin_ids), workers)]

    # Batch CPUs so we can pin workers to them with taskset linux command.
    # Note: pretty sure it's something about scikit-learn that makes this 
    # necessary, or the interaction of scikit-learn's use of multiprocessing, 
    # combined with our use of multiprocessing and/or joblib threading on 
    # top of all of that. If we don't do it this way, every worker tries to 
    # use all of the CPUs.
    CPU_batches = batch_cpus(workers)
    
    # Holder for results
    parsed_results = {}

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Loop on the bins
    for bin_batch in bin_batches:

        # Instantiate worker pool.
        pool = mp.Pool(
            processes = workers,
            maxtasksperchild = 1
        )

        # Holder for worker returns
        async_results = []

        for bin_id, bin_CPUs in zip(bin_batch, CPU_batches):

            print(f'Running optimization on {bin_id} with {len(bin_CPUs)} threads, CPUs: {", ".join(bin_CPUs)}')

            # Pull the training features for this bin
            bin_training_features_df = data_lake[f'training/{bin_id}/features']

            # Pull the training labels for this bin
            bin_training_labels = data_lake[f'training/{bin_id}/labels']

            async_results.append(
                pool.apply_async(hyperparameter_tune_bin,
                    args = (
                        bin_training_features_df, 
                        bin_training_labels,
                        parameter_distributions,
                        scoring_funcs,
                        cv_folds,
                        n_iterations,
                        bin_CPUs,
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

        # Parse the results
        for result in results:

            bin_id = result[0]
            best_parameters = result[1]
            parsed_results[bin_id] = best_parameters

    data_lake.close()

    return parsed_results


def hyperparameter_tune_bin(
        features_df: pd.DataFrame, 
        labels: pd.Series,
        parameter_distributions: dict,
        scoring_funcs: dict,
        cv_folds: int,
        n_iterations: int, 
        bin_CPUs: list[str],
        bin_id: str,
        shuffle_control: bool
) -> tuple[str, dict]:
    
    '''Runs hyperparameter tuning on bin data with XGBClassifier'''
    
    # Assign CPUs to this process. Note: pretty sure it's something
    # about scikit-learn that makes this necessary, or the interaction
    # of scikit-learn's use of multiprocessing, combined with our use
    # of multiprocessing and/or joblib threading on top of all of that.
    # If we don't do it this way, every worker tries to use all of the
    # CPUs.
    pid = os.getpid()
    cmd = 'taskset -cp %s %i' % (','.join(bin_CPUs), pid)
    _ = os.popen(cmd).read()

    # Clean up the features for training
    features = prep_data(features_df, shuffle_control)

    # Use threading backend for parallelism because we are running in a
    # daemonic worker process started by multiprocessing and thus can't 
    # re-use multiprocessing (via scikit-learn) again to spawn more processes
    with parallel_backend('threading', n_jobs = len(bin_CPUs)):

        # Instantiate the classifier
        model = XGBClassifier()

        # Set-up the hyperparameter search
        classifier = RandomizedSearchCV(
            model,
            parameter_distributions,
            scoring = scoring_funcs,
            cv = cv_folds,
            refit = 'negated_binary_cross_entropy',
            n_jobs = len(bin_CPUs),
            return_train_score = True,
            n_iter = n_iterations
        )

        # Run the hyperparameter search
        results = classifier.fit(features, labels)

    return bin_id, results


def parse_hyperparameter_tuning_results(results: dict) -> tuple[dict, pd.DataFrame]:
    '''Takes results dictionary from hyperparameter_tune_bins(). Returns
    cross-validation results as Pandas dataframe and the winning parameters and
    models in a dictionary where the bin id is the key.'''

    # Holders for parsed results
    cv_results = []
    winners = {}

    # Loop on the bins and plot the results for each
    for bin_id, result in results.items():
        print(f'{bin_id} best score: {result.best_score_}')

        # Collect the best estimator and it's parameters for each bin
        winners[bin_id] = {}
        winners[bin_id]['model'] = result.best_estimator_
        winners[bin_id]['parameters'] = result.best_params_

        # Get the cross-validation results for this bin as a dataframe
        cv_result = pd.DataFrame(result.cv_results_)

        # Add a column for the bin ID
        cv_result ['bin'] = [bin_id] * len(cv_result)

        # Add the dataframe for this bin to the results
        cv_results.append(cv_result)

    # Combine the cross-validation dataframe
    cv_results = pd.concat(cv_results)

    return winners, cv_results


def add_winners_to_parsed_results(cv_results: dict, parsed_results: dict, cv_folds: int = 5) -> dict:
    '''Takes cross-validation results from parse_hyperparameter_tuning_results()
    in parallel_xgboost functions and parsed results from baseline cross-validation.
    Adds cross-validation results from winning hyperparameter tuning model for
    each bin and returns updated parsed results for plotting.'''

    # Loop on the bins
    for bin_id in cv_results['bin'].unique():

        # Add the fold numbers
        parsed_results['Fold'].extend(list(range(cv_folds)))

        # Add the bin id
        parsed_results['Condition'].extend([bin_id] * cv_folds)

        # Add the optimization
        parsed_results['Optimized'].extend([True] * cv_folds)

        # Get the data for this bin
        bin_data = cv_results[cv_results['bin'] == bin_id]

        # Get the best ranked condition by negated binary cross entropy
        bin_winner = bin_data[bin_data['rank_test_negated_binary_cross_entropy'] == 1]
        bin_winner.reset_index(inplace = True, drop = True)

        # Next we have to get the metric scores for each split. This is kind of a pain.

        # Swap axes
        bin_winner_long = bin_winner.transpose()
        bin_winner_long.reset_index(inplace = True)

        # Dictionary to translate names in our parsed results data struct to variables in the hyperparameter optimization output
        metrics = {
            'Accuracy (%)': 'test_accuracy', 
            'False positive rate': 'test_false_positive_rate', 
            'False negative rate': 'test_false_negative_rate', 
            'Binary cross-entropy': 'test_binary_cross_entropy'
        }

        for name, key in metrics.items():
            for i in range(cv_folds):
                value = bin_winner_long[0][bin_winner_long['index'] == f'split{i}_{key}']
                parsed_results[name].extend(value)

    return parsed_results

def batch_cpus(workers: int) -> list[list[str]]:
    '''For use with scikit-learn multithreading within multiprocessing worker.
    Takes worker count and determines how many CPUs to give each worker.
    Returns list of lists containing cpu indices to assign each worker.'''

    # Figure out how many CPUs we can afford to give each worker
    # using two less than the system total
    threads_per_worker = (mp.cpu_count() - 2) // workers
    print(f'Will assign each worker {threads_per_worker} CPUs')
    
    # List of CPUs indices to be assigned 
    CPUs = list(range(mp.cpu_count() - 2))

    # Convert CPU indices to string
    CPUs = [str(i) for i in CPUs]

    # Batch CPUs
    CPU_batches = [CPUs[i:i + threads_per_worker] for i in range(0, len(CPUs), threads_per_worker)]

    return CPU_batches