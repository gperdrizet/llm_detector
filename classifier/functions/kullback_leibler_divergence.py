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
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.stats import gaussian_kde

import functions.helper as helper_funcs
import functions.multiprocess_logging as log_funcs
import configuration as config


def get_kullback_leibler_KDEs(
        feature_name: str,
        hdf5_file: str,
        logfile_name: str = 'kld.log'
) -> None:

    '''Main function to parallelize computation of Kullback-Leibler kernel density estimate
    over length bins. Saves KDE to disk.'''

    # Set-up logging
    logfile = f'{config.LOG_PATH}/{logfile_name}'
    print(f'Will log to: {logfile}\n')

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
    # CPU (minus 2 so we don't lock up) or the humber of bins
    n_workers = min(mp.cpu_count() - 2, len(list(bins.keys())))

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

        # Remove outliers with extreme perplexity or cross perplexity scores


        async_results.append(
            pool.apply_async(get_kullback_leibler_KDE,
                args = (
                    feature_name,
                    bin_training_features_df, 
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
    data_lake.close()

    # Get the results
    new_results = [async_result.get() for async_result in async_results]

    # Check result status
    for new_result in new_results:

        # Parse the result
        bin_id = new_result[0]
        finished = new_result[1]

        # Print status
        print(f'{bin_id} finished: {finished}')


def get_kullback_leibler_KDE(
        feature_name: str,
        bin_training_features_df: pd.DataFrame, 
        worker_num: int,
        bin_id: str,
        logging_queue: Callable,
        configure_logging: Callable,
) -> tuple[str, bool]:

    '''Takes feature name, training and testing features dataframes and calculates
    Kullback-Leibler divergence score for the specified feature based on training
    data. Scores each fragment in training and testing data, adds result as new 
    feature column and returns updated dataframes.'''

    # Set-up logging
    configure_logging(logging_queue)
    logger = logging.getLogger(f'{__name__}.get_kullback_leibler_KDE')
    logger.info(f'Worker {worker_num} - {len(bin_training_features_df)} fragments in {bin_id}')

    # Get a kernel density estimate for the feature's distribution in the human and synthetic data
    try:
        start_time = time.time()
        human_feature_kde, synthetic_feature_kde = get_feature_kdes(bin_training_features_df, feature_name)
        logger.info(f'Worker {worker_num} - get_feature_kdes() took {(time.time() - start_time):.3f} seconds')

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_feature_kdes() error: {err_string}')

    # Calculate the Kullback-Leibler divergence between the human and synthetic kernel density estimates
    try:
        start_time = time.time()

        feature_kld, x = get_feature_kld(
            bin_training_features_df,
            feature_name,
            human_feature_kde, 
            synthetic_feature_kde
        )

        logger.info(f'Worker {worker_num} - get_feature_kld() took {(time.time() - start_time):.3f} seconds')

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_feature_kld() error: {err_string}')
    
    # Get a kernel density estimate of the Kullback-Leibler divergence and save to disk
    try:

        start_time = time.time()
        kld_kde = get_kld_kde(feature_kld, x)
        logger.info(f'Worker {worker_num} - get_kld_kde() took {(time.time() - start_time):.3f} seconds')

        # Make a filename for the KDE
        formatted_feature_name = feature_name.replace(' ', '_').lower()
        output_filename = f'{config.MODELS_PATH}/{formatted_feature_name}_KLD_KDE_{bin_id}.pkl'

        # Pickle the KDE to disk for later use
        with open(output_filename, 'wb') as output_file:
            pickle.dump(kld_kde, output_file)

        logger.info(f'Worker {worker_num} - Kullback-Leibler divergence kernel density estimate saved to disk: {output_filename}')
    
    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_kld_kde() error: {err_string}')

    return bin_id, True

    # # Calculate Kullback-Leibler scores for the training and testing data
    # # in this bin and add the to the features
    # try:
    #     logger.info(f'Worker {worker_num} - adding Kullback-Leibler score to training features')
    #     start_time = time.time()
    #     bin_training_features_df = add_kld_score(bin_training_features_df, feature_name, feature_scaler, kld_kde)
    #     logger.info(f'Worker {worker_num} - add_kld_score(), training data took {(time.time() - start_time):.3f} seconds')

    #     logger.info(f'Worker {worker_num} - adding Kullback-Leibler score to testing features')
    #     start_time = time.time()
    #     bin_testing_features_df = add_kld_score(bin_testing_features_df, feature_name, feature_scaler, kld_kde)
    #     logger.info(f'Worker {worker_num} - add_kld_score(), testing data took {(time.time() - start_time):.3f} seconds')

    # except Exception as err_string:
    #     logger.error(f'Worker {worker_num} - add_kld_score() error: {err_string}')

    # return bin_id, bin_training_features_df, bin_testing_features_df


def get_feature_kdes(
        data_df: pd.DataFrame, 
        feature_name: str
) -> tuple[gaussian_kde, gaussian_kde]:
    
    '''Takes Pandas dataframe and a feature name. Splits data by text 'Source'
    feature. Gets kernel density estimates of distributions of feature data for 
    human and synthetic text. Returns KDEs.'''

    logger = logging.getLogger(f'{__name__}.get_feature_kdes')

    # Get caller specified data for human and synthetic text fragments
    human_data = np.asarray(data_df[feature_name][data_df['Source'] == 'human'])
    synthetic_data = np.asarray(data_df[feature_name][data_df['Source'] == 'synthetic'])

    # Clip outliers above or below 4 sigma from the mean
    human_data = helper_funcs.sigma_clip_data(data = human_data, n_sigma = 4.0)
    synthetic_data = helper_funcs.sigma_clip_data(data = synthetic_data, n_sigma = 4.0)

    # Get gaussian KDEs using SciPy and the Silverman rule-of-thumb for bandwidth estimation
    start_time = time.time()
    human_feature_kde = gaussian_kde(human_data.flatten(), bw_method = 'silverman')
    synthetic_feature_kde = gaussian_kde(synthetic_data.flatten(), bw_method = 'silverman')
    logger.debug(f'KDEs took {(time.time() - start_time):.3f} seconds')

    return human_feature_kde, synthetic_feature_kde


def kl_divergence(p: list, q: list) -> np.ndarray:
    '''Takes two lists, calculates Kullback-Leibler divergence.'''

    # Convert inputs to numpy
    p = np.asarray(p)
    q = np.asarray(q)

    # Set handling for overflows/underflows - just ignore. We will handle infinite 
    # or nan values later by just filtering them out.
    with np.errstate(over = 'ignore', under = 'ignore', divide = 'ignore', invalid='ignore'):

        kld_values = p * np.log2(p/q)

    return kld_values


def get_feature_kld(
        data_df: pd.DataFrame,
        feature_name: str,
        human_feature_kde: gaussian_kde, 
        synthetic_feature_kde: gaussian_kde
) -> tuple[np.ndarray, np.ndarray]:
    
    '''Takes kernel density estimates of data distributions for human and 
    synthetic data and original dataset as dataframe. Calculates Kullback-Leibler 
    divergences of distributions at set of regularly spaced sample points covering 
    the data's range plus some padding on either edge. Returns the Kullback-Leibler 
    divergence values and the sample points used to calculate them.'''

    logger = logging.getLogger(f'{__name__}.get_feature_kld')

    # Get feature data
    scores = np.array(data_df[feature_name])

    # Get a list of points covering the range of score values and extend
    # the left and right edges a little bit, otherwise the kernel density
    # estimate tends to droop at the edges of the range. We will clip
    # the padding off later.
    x = helper_funcs.make_padded_range(scores)

    logger.debug(f'Num samples: {x.shape}')

    # Get fitted values for the points
    human_fitted_values = human_feature_kde(x)
    synthetic_fitted_values = synthetic_feature_kde(x)

    # Calculate the KL divergences of the fitted values
    kld = kl_divergence(synthetic_fitted_values, human_fitted_values)

    # Get rid of any np.nan, without changing the length
    mask = np.isnan(kld)
    kld[mask] = 0

    # Get rid of any inf without changing the length
    mask = np.isinf(kld)
    kld[mask] = 0

    return kld, x


def get_kld_kde(kld: np.ndarray, x: np.ndarray) -> gaussian_kde:
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
    kld_kde = gaussian_kde(np.array(kld_scores).flatten(), bw_method = 'silverman')

    return kld_kde


def add_kld_score(
        data_df: pd.DataFrame, 
        feature_name: str,
        feature_scaler: gaussian_kde,
        kld_kde: gaussian_kde
) -> pd.DataFrame:
    
    '''Takes a features dataframe, feature name and Kullback-Leibler kernel density estimate,
    calculates a Kullback-Leibler score from each text fragment's value for the named feature
    and adds it back to the dataframe as a new feature using the caller specified feature name
    with 'Kullback-Leibler divergence' appended as the new column name'''

    logger = logging.getLogger(f'{__name__}.add_kld_score')

    # Calculate the KLD scores
    start_time = time.time()
    scores = feature_scaler.transform(np.asarray(data_df[feature_name]).reshape(-1, 1))
    kld_scores = kld_kde.score_samples(np.asarray(scores).reshape(-1, 1))
    logger.debug(f'KLD PDF calculation took {round(time.time() - start_time, 1)} sec.')

    # Add the scores back to the dataframe in a new column
    start_time = time.time()
    data_df[f'{feature_name} Kullback-Leibler divergence'] = kld_scores
    logger.debug(f'KLD feature creation took {round(time.time() - start_time, 1)} sec.')

    return data_df


def parallel_evaluate_scores(kde: gaussian_kde, data: np.array, workers: int):
    '''Splits evaluation over n_workers. Meant to be called by make_kullback_leibler_feature().'''

    with mp.Pool(workers) as p:
        return np.concatenate(p.map(kde, np.array_split(data, workers)))


def make_kullback_leibler_feature(
        feature_name: str,
        hdf5_file: str,
        logfile_name: str = 'kld.log'
) -> None:
    
    '''uses previously stored kernel density estimates of the Kullback-Leibler 
    divergence between a feature score for human and synthetic text fragments
    in each bin. Loads the data from each bin sequentially and evaluates
    each text fragment's perplexity ratio score. Adds result as new
    feature column. Parallelizes evaluation within each bin.'''

    # Set-up logging
    logger = helper_funcs.start_logger(
        logfile_name = logfile_name,
        logger_name = __name__
    )

    # Get the bins from the hdf5 file's metadata
    data_lake = h5py.File(hdf5_file, 'r')
    bins = dict(data_lake.attrs.items())
    data_lake.close()

    # Open a connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(hdf5_file)

    # Loop on the bins
    for bin_id in bins.keys():

        logger.info(f'Evaluating KLD in bin {bin_id}')

        # Pull the training features for this bin
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Pull the testing features for this bin
        bin_testing_features_df = data_lake[f'testing/{bin_id}/features']

        # Make the filename for the stored Kullback-Leibler divergence kernel density estimate for this bin
        formatted_feature_name = feature_name.replace(' ', '_').lower()
        input_filename = f'{config.MODELS_PATH}/{formatted_feature_name}_KLD_KDE_{bin_id}.pkl'

        # Load the KLD KDE
        with open(input_filename, 'rb') as input_file:
            kld_kde = pickle.load(input_file)

        # Evaluate the feature scores
        start_time = time.time()
        training_kde_values = parallel_evaluate_scores(
            kde = kld_kde,
            data = np.array(bin_training_features_df[feature_name]),
            workers = mp.cpu_count() - 2
        )
        logger.debug(f'KLD {bin_id} training evaluation took {round(time.time() - start_time, 1)} sec.')

        # Add the scores back to the dataframe in a new column
        start_time = time.time()
        bin_training_features_df[f'{feature_name} Kullback-Leibler divergence'] = training_kde_values
        logger.debug(f'KLD {bin_id} training feature creation took {round(time.time() - start_time, 1)} sec.')

        start_time = time.time()
        testing_kde_values = parallel_evaluate_scores(
            kde = kld_kde,
            data = np.array(bin_testing_features_df[feature_name]),
            workers = mp.cpu_count() - 2
        )
        logger.debug(f'KLD {bin_id} testing evaluation took {round(time.time() - start_time, 1)} sec.')

        # Add the scores back to the dataframe in a new column
        start_time = time.time()
        bin_testing_features_df[f'{feature_name} Kullback-Leibler divergence'] = testing_kde_values
        logger.debug(f'KLD {bin_id} testing feature creation took {round(time.time() - start_time, 1)} sec.')

        # Put data back into hdf5
        data_lake.put(f'training/{bin_id}/features', bin_training_features_df)
        data_lake.put(f'testing/{bin_id}/features', bin_testing_features_df)

        print(f'{bin_id} finished: True')

    data_lake.close()


def plot_results(hdf5_file: str, feature_name: str) -> plt:
    '''Takes feature name and hdf5 file with Kullback-Leibler score for that
    feature, plots some diagnostic distributions for each bin.'''

    # Get bins from hdf5 metadata
    data_lake = h5py.File(hdf5_file, 'r')
    bins = dict(data_lake.attrs.items())
    bin_ids = list(bins.keys())
    data_lake.close()

    # Open a connection to the hdf5 dataset via PyTables with Pandas so we can
    # load the data from each bin as a dataframe
    data_lake = pd.HDFStore(hdf5_file)

    # Now we want to make 3 plots for each bin: the distributions of perplexity ratio score,
    # the Kullback-Leibler divergence kernel density estimate, and the distribution of 
    # Kullback-Leibler score values

    # Set up a figure for n bins x 3 plots
    fig, axs = plt.subplots(
        len(bin_ids),
        3,
        figsize = (9, (3 * len(bin_ids))),
        gridspec_kw = {'wspace':0.4, 'hspace':0.4},
        #sharex='col'
    )

    # Loop on the bins to draw each plot
    for i, bin_id in enumerate(bin_ids):

        # Load bin data
        bin_training_features_df = data_lake[f'training/{bin_id}/features']

        # Get human and synthetic perplexity ratio score
        human_feature = bin_training_features_df[feature_name][bin_training_features_df['Source'] == 'human']
        synthetic_feature = bin_training_features_df[feature_name][bin_training_features_df['Source'] == 'synthetic']

        # Draw histograms for human and synthetic perplexity ratio scores in the first plot
        axs[i, 0].set_title(f'{bin_id} {feature_name.lower()}', fontsize = 'medium')
        axs[i, 0].set_xlabel('Score')
        axs[i, 0].set_ylabel('Count')
        axs[i, 0].hist(human_feature, bins = 50, alpha = 0.5, label = 'human')
        axs[i, 0].hist(synthetic_feature, bins = 50, alpha = 0.5, label = 'synthetic')
        axs[i, 0].legend(loc = 'upper left', fontsize = 'x-small')

        # Turn axis tick labels back on for shared x axis
        axs[i, 0].tick_params(labelbottom = True)

        # For the second plot load and evaluate the Kullback-Leibler divergence kernel density estimate
            
        # Make the filename for the stored Kullback-Leibler divergence kernel density estimate for this bin
        formatted_feature_name = feature_name.replace(' ', '_').lower()
        input_filename = f'{config.MODELS_PATH}/{formatted_feature_name}_KLD_KDE_{bin_id}.pkl'

        # Load the KLD KDE
        with open(input_filename, 'rb') as input_file:
            kld_kde = pickle.load(input_file)

        # Make 100 evaluation points across the data's range
        x = helper_funcs.make_padded_range(bin_training_features_df[feature_name])

        # Evaluate
        y = kld_kde(x)

        # Clip data back to original range
        x_clipped = []
        y_clipped = []

        for xi, yi in zip(x, y):
            if xi >= min(bin_training_features_df[feature_name]) and xi <= max(bin_training_features_df[feature_name]):
                x_clipped.append(xi)
                y_clipped.append(yi)

        # Plot
        axs[i, 1].set_title(f'{bin_id} Kullback-Leibler KDE', fontsize = 'medium')
        axs[i, 1].set_xlabel('Score')
        axs[i, 1].set_ylabel('KDE value')
        axs[i, 1].plot(x_clipped, y_clipped)

        # Turn axis tick labels back on for shared x axis
        axs[i, 1].tick_params(labelbottom = True)

        # For the third plot make a histogram of the KLD KDE values in the bin

        # Get human and KLD scores
        human_feature = bin_training_features_df[f'{feature_name} Kullback-Leibler divergence'][bin_training_features_df['Source'] == 'human']
        synthetic_feature = bin_training_features_df[f'{feature_name} Kullback-Leibler divergence'][bin_training_features_df['Source'] == 'synthetic']

        # Draw histograms for human and synthetic Kullback-Leibler divergence scores
        axs[i, 2].set_title(f'{bin_id} Kullback-Leibler scores', fontsize = 'medium')
        axs[i, 2].set_xlabel('Score')
        axs[i, 2].set_ylabel('Count')
        axs[i, 2].hist(human_feature, bins = 50, alpha = 0.5, label = 'human')
        axs[i, 2].hist(synthetic_feature, bins = 50, alpha = 0.5, label = 'synthetic')
        axs[i, 2].legend(loc = 'upper left', fontsize = 'x-small')

        # Turn axis tick labels back on for shared x axis
        axs[i, 2].tick_params(labelbottom = True)

    data_lake.close()

    return plt