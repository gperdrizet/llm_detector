'''Set of functions to calculate Kullback-Leibler divergence divergence between
human and synthetic data for a caller specified feature. Uses gaussian kernel
density estimate of Kullback-Leibler divergence distribution for feature to
calculate 'Kullback-Leibler score' for each text fragment and add it back to
the data as a new feature.'''

from __future__ import annotations
from typing import Callable

import h5py
import time
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.neighbors import KernelDensity

import functions.multiprocess_logging as log_funcs
import configuration as config


def kullback_leibler_score(
        feature_name: str,
        hdf5_file: str,
        score_sample: bool = False,
        logfile_name: str = 'kld.log'
) -> None:

    '''Main function to parallelize computation of Kullback-Leibler score
    over length bins.'''

    # Set-up logging
    logfile = f'{config.LOG_PATH}/{logfile_name}'
    print(f'Will log to: {logfile}')

    logging_queue = mp.Manager().Queue(-1)

    log_listener = mp.Process(
        target = log_funcs.listener_process,
        args = (logging_queue, log_funcs.configure_listener, logfile)
    )

    log_listener.start()

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(hdf5_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Calculate worker number whichever is less, the number of avalible
    # CPU or the humber of bins
    n_workers = min(20, len(list(bins.keys())))

    # Instantiate worker pool
    pool = mp.Pool(
        processes = n_workers,
        maxtasksperchild = 1
    )

    # Holder for returns from workers
    async_results = []

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(hdf5_file)

    # Loop on the bins
    for worker_num, bin_id in enumerate(bins.keys()):

        # Pull the training features for this bin
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Pull the testing features for this bin
        bin_testing_features_df = data_lake[f'testing/{bin_id}/features']

        # Take sample if desired
        if score_sample is True:
            bin_training_features_df = bin_training_features_df.sample(frac = 0.1)
            bin_testing_features_df = bin_testing_features_df.sample(frac = 0.1)

        async_results.append(
            pool.apply_async(add_feature_kld_score,
                args = (
                    feature_name,
                    bin_training_features_df, 
                    bin_testing_features_df,
                    worker_num,
                    bin_id,
                    logging_queue, 
                    log_funcs.configure_worker
                )
            )
        )

    # Clean up
    pool.close()
    pool.join()

    logging_queue.put_nowait(None)
    log_listener.join()

    # Get the results
    new_results = [async_result.get() for async_result in async_results]

    # Add the new results
    for new_result in new_results:

        # Parse the result
        bin_id = new_result[0]
        training_features_df = new_result[1]
        testing_features_df = new_result[2]

        # Put data back into hdf5
        data_lake.put(f'training/{bin_id}/features', training_features_df)
        data_lake.put(f'testing/{bin_id}/features', testing_features_df)

    data_lake.close()


def add_feature_kld_score(
        feature_name: str,
        bin_training_features_df: pd.DataFrame, 
        bin_testing_features_df: pd.DataFrame,
        worker_num: int,
        bin_id: str,
        logging_queue: Callable,
        configure_logging: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Takes feature name, training and testing features dataframes and calculates
    Kullback-Leibler divergence score for the specified feature based on training
    data. Scores each fragment in training and testing data, adds result as new 
    feature column and returns updated dataframes.'''

    # Set-up logging
    configure_logging(logging_queue)
    logger = logging.getLogger(f'{__name__}.add_feature_kld_score')
    logger.info(f'Worker {worker_num} - {len(bin_training_features_df)} fragments in {bin_id}')

    # Calculate the feature's distribution kernel density estimates
    try:
        start_time = time.time()
        human_feature_kde, synthetic_feature_kde = get_kdes(bin_training_features_df, feature_name)
        logger.info(f'Worker {worker_num} - get_kdes() took {round(time.time() - start_time, 1)} seconds')

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_pr_score_kdes() error: {err_string}')

    try:
        # Calculate the Kullback-Leibler divergence
        start_time = time.time()

        feature_kld, x = get_kld(
            bin_training_features_df,
            feature_name,
            human_feature_kde, 
            synthetic_feature_kde
        )

        logger.info(f'Worker {worker_num} - get_kld() took {round(time.time() - start_time, 1)} seconds')

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_kld() error: {err_string}')
    
    # Calculate the Kullback-Leibler divergence of the feature's distributions
    # in the human and synthetic data
    try:

        start_time = time.time()
        kld_kde = get_kld_kde(feature_kld, x)
        logger.info(f'Worker {worker_num} - get_kld_kde() took {round(time.time() - start_time, 1)} seconds')
    
    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_kld_kde() error: {err_string}')

    # Calculate Kullback-Leibler scores for the training and testing data
    # in this bin and add the to the features
    try:
        logger.info(f'Worker {worker_num} - adding Kullback-Leibler score to training features')
        start_time = time.time()
        bin_training_features_df = add_kld_score(bin_training_features_df, feature_name, kld_kde)
        logger.info(f'Worker {worker_num} - add_kld_score(), training data took {round(time.time() - start_time, 1)} seconds')

        logger.info(f'Worker {worker_num} - adding Kullback-Leibler score to testing features')
        start_time = time.time()
        bin_testing_features_df = add_kld_score(bin_testing_features_df, feature_name, kld_kde)
        logger.info(f'Worker {worker_num} - add_kld_score(), testing data took {round(time.time() - start_time, 1)} seconds')

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - add_kld_score() error: {err_string}')

    return bin_id, bin_training_features_df, bin_testing_features_df


def get_kdes(data_df: pd.DataFrame, feature_name: str) -> tuple[KernelDensity, KernelDensity]:
    '''Takes Pandas dataframe and a feature name. Splits data by text 'Source'
    feature. Gets kernel density estimates of distributions of data specified 
    by feature name for human and synthetic text. Returns KDEs.'''

    # Get caller specified data for human and synthetic text fragments
    human_data = data_df[feature_name][data_df['Source'] == 'human']
    synthetic_data = data_df[feature_name][data_df['Source'] == 'synthetic']

    # Get KDEs
    human_feature_kde = KernelDensity(kernel = 'gaussian').fit(np.asarray(human_data).reshape(-1, 1))
    synthetic_feature_kde = KernelDensity(kernel = 'gaussian').fit(np.asarray(synthetic_data).reshape(-1, 1))

    return human_feature_kde, synthetic_feature_kde


def kl_divergence(p: list, q: list) -> np.ndarray:
    '''Takes two lists, calculates Kullback-Leibler divergence.'''

    # Convert inputs to numpy
    p = np.asarray(p)
    q = np.asarray(q)

    # Set handling for overflows/underflows - just ignore. We will handle infinite 
    # or nan values later by just filtering them out.
    with np.errstate(over = 'ignore', under = 'ignore', divide = 'ignore'):

        kld_values = p * np.log2(p/q)

    return kld_values


def get_kld(
        data_df: pd.DataFrame,
        feature_name: str,
        human_feature_kde: KernelDensity, 
        synthetic_feature_kde: KernelDensity
) -> tuple[np.ndarray, np.ndarray]:
    
    '''Takes kernel density estimates of data distributions for human and 
    synthetic data and original dataset as dataframe. Calculates Kullback-Leibler
    divergences of distributions at set of regularly spaced sample points covering
    the original data's range plus some padding on either edge. Returns the 
    Kullback-Leibler divergence values and the sample points used to calculate them.'''

    logger = logging.getLogger(f'{__name__}.get_kld')

    # Get feature data
    scores = data_df[feature_name]

    # Get a list of points covering the range of score values and extend
    # the left and right edges a little bit, otherwise the kernel density
    # estimate tends to droop at the edges of the range. We will clip
    # the padding off later.
    data_range = max(scores) - min(scores)
    logger.debug(f'Data range: {data_range}, {min(scores)}, {max(scores)}')

    padding = data_range * 0.1
    logger.debug(f'Padding: {padding}')

    x_max = max(scores) + padding
    x_min = min(scores) - padding
    logger.debug(f'Sample range: {x_max - x_min}, {x_min} - {x_max}')

    sample_frequency = (x_max - x_min) / 100
    logger.debug(f'Sample frequency: {sample_frequency}')

    x = np.arange(
        min(scores) - padding, 
        max(scores) + padding, 
        sample_frequency
    )

    logger.debug(f'Num samples: {x.shape}')

    # Get fitted values for the points
    human_fitted_values = human_feature_kde.score_samples(x.reshape(-1, 1))
    synthetic_fitted_values = synthetic_feature_kde.score_samples(x.reshape(-1, 1))

    # Calculate the KL divergences of the fitted values
    kld = kl_divergence(synthetic_fitted_values, human_fitted_values)

    # Get rid of any np.nan, without changing the length
    mask = np.isnan(kld)
    kld[mask] = 0

    # Get rid of any inf without changing the length
    mask = np.isinf(kld)
    kld[mask] = 0

    return kld, x


def get_kld_kde(kld: np.ndarray, x: np.ndarray) -> KernelDensity:
    '''Takes list of Kullback-Leibler divergence values, and regularly
    spaced sample points taken from original data's range used to generate 
    them. Generates and returns gaussian kernel density estimate. Trick 
    here is that the KLD values are 'density' as they are derived from 
    the KDEs of the PR score distributions. Therefore they need to be 
    converted back to 'raw' data.'''

    # Convert the KLD 'density' values into integer 'count' values

    # Shift the kld values so that they are non-negative
    kld = kld + abs(min(kld))

    # Then scale the values so when we convert to integer we get good
    # resolution, e.g. we don't want to collapse 2.1, 2.2, 2.3 etc.,
    # to 2. Instead, 2100.0, 2200.0, 2300.0 become 2100, 2200, 2300 etc.
    kld = kld * 1000

    # Convert to integer
    kld_counts = kld.astype(int)

    # Now, construct a list where each value of x appears a number of times
    # equal to it's KLD 'count'
    kld_scores = []

    for i in range(len(kld_counts)):
        kld_scores.extend([x[i]] * kld_counts[i])

    # Then, run a KDE on the reconstructed KLD scores
    kld_kde = KernelDensity(kernel = 'gaussian').fit(np.asarray(kld_scores).reshape(-1, 1))

    return kld_kde


def add_kld_score(data_df: pd.DataFrame, feature_name: str, kld_kde: KernelDensity) -> pd.DataFrame:
    '''Takes a features dataframe, feature name and Kullback-Leibler kernel density estimate,
    calculates a Kullback-Leibler score from each text fragment's value for the named feature
    and adds it back to the dataframe as a new feature using the caller specified feature name
    with 'Kullback-Leibler divergence' appended'''

    logger = logging.getLogger(f'{__name__}.add_kld_score')

    # Calculate the KLD scores
    start_time = time.time()
    kld_scores = kld_kde.score_samples(np.asarray(data_df[feature_name]).reshape(-1, 1))
    logger.debug(f'KLD PDF calculation took {round(time.time() - start_time, 1)} sec.')

    # Add the scores back to the dataframe in a new column
    start_time = time.time()
    data_df[f'{feature_name} Kullback-Leibler divergence'] = kld_scores
    logger.debug(f'KLD feature creation took {round(time.time() - start_time, 1)} sec.')

    return data_df