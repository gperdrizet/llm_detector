# '''Collection of functions for XGBoost classifier training/evaluation with
# length binned data. Parallelizes models over the bins.'''

from __future__ import annotations
# from typing import Callable

# import os
# import logging
import pickle
# import multiprocessing as mp

import h5py
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
# from joblib import parallel_backend

import functions.notebook_helper as helper_funcs
# import configuration as config # pylint: disable=import-error


def cross_validate_bins(
        input_file: str = None,
        parsed_results: dict = None,
        scoring_funcs: dict = None,
        cv_folds: int = 5,
        shuffle_control: bool = False,
) -> dict:
    
    'Runs cross-validation on length binned data.'
    
    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(input_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Loop on the bins
    for bin_id in bins.keys():

        print(f'Cross-validating {bin_id}')

        # Pull the training features for this bin
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Clean up the features for training
        bin_training_features_df = prep_data(
            features_df = bin_training_features_df,
            feature_drops = ['Fragment length (words)', 'Source', 'String'],
            shuffle_control = shuffle_control
        )

        # Pull the training labels for this bin
        bin_training_labels = data_lake[f'training/{bin_id}/labels']

        # Instantiate sklearn gradient boosting classifier
        model = XGBClassifier(random_state = 42)

        # Run cross-validation
        scores = cross_validate(
            model,
            bin_training_features_df.to_numpy(),
            bin_training_labels.to_numpy(),
            cv = cv_folds,
            n_jobs = -1,
            scoring = scoring_funcs
        )

        parsed_results = helper_funcs.add_two_factor_cv_scores(
            parsed_results,
            scores,
            bin_id,
            False
        )

    data_lake.close()

    return parsed_results


def hyperparameter_optimize_bins(
        input_file: str = None,
        parameter_distributions: dict = None,
        scoring_funcs: dict = None,
        cv_folds: int = 5,
        hyperparameter_iterations: int = 10
) -> dict:
    
    '''Runs hyperparameter optimization on length binned data.'''

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(input_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Empty dictionary to collect results
    results = {}

    # Loop on bins
    for bin_id in bins.keys():
        print(f'Optimizing {bin_id}')

        # Pull the training features for this bin
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Clean up the features for training
        bin_training_features_df = prep_data(
            features_df = bin_training_features_df,
            feature_drops = ['Fragment length (words)', 'Source', 'String']
        )

        # Pull the training labels for this bin
        bin_training_labels = data_lake[f'training/{bin_id}/labels']

        # Instantiate the classifier
        model = XGBClassifier()

        # Set-up the hyperparameter search
        classifier = RandomizedSearchCV(
            model,
            parameter_distributions,
            scoring = scoring_funcs,
            cv = cv_folds,
            refit = 'negated_binary_cross_entropy',
            n_jobs = -1,
            return_train_score = True,
            n_iter = hyperparameter_iterations
        )

        # Run the hyperparameter search
        result = classifier.fit(bin_training_features_df, bin_training_labels)

        # Collect the result
        results[bin_id] = result

    data_lake.close()

    return results


def prep_data(
        features_df: pd.DataFrame,
        feature_drops: list,
        shuffle_control: bool = False
) -> pd.DataFrame:

    '''Takes a dataframe of features, drops unnecessary or un-trainable features,
    cleans up missing data, shuffles and returns updated dataframe.'''

    # Get rid of un-trainable/unnecessary features
    features_df.drop(feature_drops, axis = 1, inplace = True)

    # Replace and remove string 'OOM' and 'NAN' values
    features_df.replace('NAN', np.nan, inplace = True)
    features_df.replace('OOM', np.nan, inplace = True)
    features_df.dropna(inplace = True)

    # Last, if this is a shuffle control run, randomize the order of the
    # data. This will break the correspondence between features and labels
    if shuffle_control is True:
        features_df = features_df.sample(frac = 1).reset_index(drop = True)

    return features_df


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


def add_winners_to_parsed_results(
        cv_results: dict,
        parsed_results: dict,
        cv_folds: int = 5
) -> dict:

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

        # Dictionary to translate names in our parsed results data struct to variables in
        # the hyperparameter optimization output
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


# def batch_cpus(workers: int) -> list[list[str]]:
#     '''For use with scikit-learn multithreading within multiprocessing worker.
#     Takes worker count and determines how many CPUs to give each worker.
#     Returns list of lists containing cpu indices to assign each worker.'''

#     # Figure out how many CPUs we can afford to give each worker
#     # using two less than the system total
#     threads_per_worker = (mp.cpu_count() - 2) // workers
#     print(f'Will assign each worker {threads_per_worker} CPUs')

#     # List of CPUs indices to be assigned
#     cpus = list(range(mp.cpu_count() - 2))

#     # Convert CPU indices to string
#     cpus = [str(i) for i in cpus]

#     # Batch CPUs
#     cpu_batches = [cpus[i:i + threads_per_worker] for i in range(0, len(cpus), threads_per_worker)]

#     return cpu_batches


def add_stage_one_probabilities_features(
    input_file: str,
    output_file: str,
    hyperparameter_optimization_results_filename: str
) -> pd.DataFrame:
    
    '''Takes paths to hdf5 dataset and hyperparameter optimization results.
    Adds stage one classifier class probabilities to combined data as new feature'''

    # Load the winning model from hyperparameter optimization for each bin
    with open(hyperparameter_optimization_results_filename, 'rb') as result_input_file:
        results = pickle.load(result_input_file)

    # Get the numerical bin length ranges from the hdf5 dataset's metadata
    input_data_lake = h5py.File(input_file, 'r')
    bins = dict(input_data_lake.attrs.items())
    input_data_lake.close()

    # Remove the combined bin - we don't want to add that score and all
    # fragments fall in the combined bin
    _ = bins.pop('combined', None)

    # Make a new hdf5 file for output
    output_data_lake = h5py.File(output_file, 'a')

    # Create the top-level groups
    _ = output_data_lake.require_group('training')
    _ = output_data_lake.require_group('testing')

    # Add a combined group under testing and training. This is where
    # we will put the new data with the class probabilities added
    _ = output_data_lake.require_group('training/combined')
    _ = output_data_lake.require_group('testing/combined')

    output_data_lake.close()

    # Next, copy the labels directly from the old dataset to the new dataset
    input_data_lake = pd.HDFStore(input_file)
    output_data_lake = pd.HDFStore(output_file)

    output_data_lake.put(f'training/combined/labels', input_data_lake['training/combined/labels'])
    output_data_lake.put(f'testing/combined/labels', input_data_lake['testing/combined/labels'])

    # Now, we need to visit each text fragment in the training and testing
    # data and score it with both classifiers that it falls into the
    # length range for
    for dataset in ['training/combined/features', 'testing/combined/features']:

        # Load the combined features
        features_df = input_data_lake[dataset]

        # Clean up the features
        features_df.replace('NAN', np.nan, inplace = True)
        features_df.replace('OOM', np.nan, inplace = True)
        features_df.dropna(inplace = True)

        # Run apply on rows to score the fragments
        features_df = features_df.apply(
            class_probability_scores,
            bins = bins,
            results = results,
            axis = 1
        )

        # Put data into the output data lake
        output_data_lake.put(dataset, features_df)

    input_data_lake.close()
    output_data_lake.close()

    return True


def class_probability_scores(
        fragment: np.ndarray,
        bins: dict = None,
        results: dict = None
) -> tuple[float, float]:

    '''Takes bin dictionary and hyperparameter tuning results.
    Uses best model for each applicable length bin to score
    text fragments.'''

    # Get the fragment length in words
    fragment_length = fragment['Fragment length (words)']

    # Next we have to loop on the bins and see which one(s)
    # this fragment falls into. Then we use the model for those
    # bins to score the fragment. Most fragments will get scored
    # by two models 'short_score' and 'long_score' while some
    # very short or very long fragments will only have one score.
    # To deal with this, let's initialize the scores to np.nan
    # and then replace them with model outputs.

    score_count = 0
    scores = [np.nan, np.nan]

    # Loop on bins
    for bin_id, bin_range in bins.items():

        # Check if this fragment goes in this bin
        if fragment_length >= bin_range[0] and fragment_length <= bin_range[1]:

            # Predict the class probability with the model for this bin
            # making sure to clip the fragment length in words
            # off of the features because it was not present for
            # model training (only length in tokens was)
            bin_model = results[bin_id].best_estimator_
            features = fragment.drop(['Fragment length (words)', 'Source', 'String'])
            class_probabilities = bin_model.predict_proba([features])

            # Get the class 0 probability
            class_probability = class_probabilities[0][0]

            # Add it to scores
            scores[score_count] = class_probability
            score_count += 1

            # If this was the second bin we got a score from, we are done
            if score_count == 2:
                break

    # Add the new features to the fragment
    fragment['Short score'] = scores[0]
    fragment['Long score'] = scores[1]

    return fragment
