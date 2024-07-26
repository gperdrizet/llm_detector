'''Collection of top level functions for Luigi tasks in the 
data & feature engineering pipeline'''

from __future__ import annotations

import gc
import pickle
import json
import multiprocessing
#from math import log2

import numpy as np
import pandas as pd
from scipy.stats import fit, exponnorm, gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

import classifier.functions.helper as helper_funcs
import classifier.configuration as config

def load_data() -> dict:
    '''Parses and combines perplexity ratio scored text fragments from
    all three Hans 2024 datasets. Returns results as training testing
    dict of JSON.'''

    # Holder for each parsed dataset
    dataframes = []

    for _, filename in config.SCORED_HANS_DATASETS.items():

        # Load the data
        dataframe = pd.read_json(f'{config.HANS_DATA_PATH}/{filename}')

        # Update name of some columns, some scoring runs used the earlier names
        dataframe.rename(columns = {
            'Binoculars score': 'Perplexity ratio score',
            'Observer peak memory (GB)': 'Reader peak memory (GB)',
            'Performer peak memory (GB)': 'Writer peak memory (GB)'
        }, inplace = True)

        # get rid of some unnecessary columns
        dataframe.drop([
            'Fragment', 
            'Reader peak memory (GB)', 
            'Writer peak memory (GB)'
        ], axis = 1, inplace = True)

        # Replace and remove string 'OOM' and 'NAN' values
        dataframe = helper_funcs.clean_ooms(dataframe)

        # Fix some d-types
        dataframe = helper_funcs.fix_dtypes(dataframe)

        # Add this dataframe to the list
        dataframes.append(dataframe)

    # Combine the individual datasets and reset the index
    data_df = pd.concat(dataframes, axis = 0)
    data_df.reset_index(inplace = True, drop = True)

    # Split the data in to training and testing subsets.
    training_df = data_df.sample(frac = config.TRAIN_TEST_SPLIT, random_state = 42)
    testing_df = data_df.drop(training_df.index)
    training_df.reset_index(inplace = True, drop = True)
    testing_df.reset_index(inplace = True, drop = True)

    # Construct a single dictionary containing the JSON of the training
    # and testing dataframes
    results = {
        'training': training_df.to_json(),
        'testing': testing_df.to_json()
    }

    return results

def perplexity_ratio_kld_kde() -> gaussian_kde:
    '''Makes gaussian kernel density estimate of the Kullback-Leibler divergence
    between the perplexity ratio score distributions of human and synthetic
    text fragments in the scored Hans data. Returns KDE model.'''

    # Load the data
    with open(config.LOADED_DATA, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    # Pull out just the training data
    training_data_df = pd.DataFrame.from_dict(json.loads(data['training']))

    # Replace and remove string 'OOM' and 'NAN' values
    training_data_df = helper_funcs.clean_ooms(training_data_df)

    # Fix some d-types
    training_data_df = helper_funcs.fix_dtypes(training_data_df)

    # Get a list of points covering the range of score values
    scores = training_data_df['Perplexity ratio score']
    x = np.arange(min(scores) - 0.25, max(scores) + 0.25, 0.001).tolist()

    # Do the exponential gaussian fits on the human and synthetic
    # scores and get fit values for f(x)

    # Seperate human and synnthetic perplexity ratio scores
    human = training_data_df[training_data_df['Source'] == 'human']
    synthetic = training_data_df[training_data_df['Source'] == 'synthetic']
    human_scores = human['Perplexity ratio score']
    synthetic_scores = synthetic['Perplexity ratio score']

    # Set approximate bounds for fit
    bounds = [[0.0,2.0],[0.0,2.0],[0.0,1.0]]

    # Do the fit on the human data
    human_exponnorm = fit(exponnorm, human_scores, bounds = bounds)

    # Use the fitted parameters to calculate values 
    human_exponnorm_fit = exponnorm(
        human_exponnorm.params.K,
        human_exponnorm.params.loc,
        human_exponnorm.params.scale
    ).pdf(x)

    # Do the fit on the synthetic data
    synthetic_exponnorm = fit(exponnorm, synthetic_scores, bounds = bounds)
    
    # Use the fitted parameters to calculate values
    synthetic_exponnorm_fit = exponnorm(
        synthetic_exponnorm.params.K,
        synthetic_exponnorm.params.loc,
        synthetic_exponnorm.params.scale
    ).pdf(x)

    # Get the Kulback-Leibler divergence of the fitted values
    kl = helper_funcs.kl_divergence(synthetic_exponnorm_fit, human_exponnorm_fit)

    # Convert the kl values into integer 'count' values
    kl = kl + abs(min(kl))
    kl = kl * 100
    kl_counts = [int(density) for density in kl]

    # Now, construct a list where each value of x appears a number of times
    # equal to it's kl 'count'
    kl_scores = []

    for i, _ in enumerate(kl_counts):
        kl_scores.extend([x[i]] * kl_counts[i])

    # Finally, run a KDE on the reconstructed KL scores
    kl_kde = gaussian_kde(kl_scores)

    return kl_kde

def add_perplexity_ratio_kld_score() -> pd.DataFrame:
    '''Uses multiprocessing to split text fragment data
    and add perpleixty ratio Kullback-Leibler divergence
    score to the chunks in parallel. Returns concatenated
    dataframes as training testing dict of JSON.'''

    # Load the data
    with open(config.LOADED_DATA, 'r', encoding='utf-8') as input_file:
        datasets = json.load(input_file)

    # Load the Kullback-Leibler divergence kernel density estimate
    with open(config.PERPLEXITY_RATIO_KLD_KDE, 'rb') as input_file:
        kl_kde = pickle.load(input_file)

    # Empty dict for results
    results = {}

    # Loop on the training & testing datsets
    for dataset, data in datasets.items():

        data_df = pd.DataFrame.from_dict(json.loads(data))

        # Split the data up for n workers
        data_chunks = np.array_split(data_df, config.KL_SCORE_WORKERS)

        # Start multiprocessing manager and use it to create and empty list
        # in shard memory to get data back from workers
        manager = multiprocessing.Manager()
        return_list = manager.list()
        
        # Loop on data chunks, submitting each for processing
        jobs = []

        for data_chunk in data_chunks:
            p = multiprocessing.Process(target = helper_funcs.add_kl_divergence_score, args=(data_chunk, kl_kde, return_list))
            jobs.append(p)
            p.start()

        # Call join on each worker in the jobs list
        for proc in jobs:
            proc.join()

        # Concatenate the return chunks from the shared memory list into a single dataframe
        # and add it to the result as json
        result = pd.concat(return_list, axis = 0)
        result.reset_index(inplace = True, drop = True)
        print(result.head())
        print()
        print(result.info())
        print(result.columns)
        results[dataset] = result.to_json()

    return results


def tfidf_score_kld_kde() -> gaussian_kde:
    '''Makes gaussian kernel density estimate of the Kullback-Leibler divergence
    between the tffidf score distributions of human and synthetic
    text fragments in the scored Hans data. Returns KDE model.'''

    # Load the data
    with open(config.PERPLEXITY_RATIO_KLD_SCORE_ADDED, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    # Pull out just the training data
    training_data_df = pd.DataFrame.from_dict(json.loads(data['training']))

    print(training_data_df.info())

    # Replace and remove string 'OOM' and 'NAN' values
    training_data_df = helper_funcs.clean_ooms(training_data_df)

    # Fix some d-types
    training_data_df = helper_funcs.fix_dtypes(training_data_df)

    # Take samples so we dont' OOM during vectorization for TF-IDF
    training_data_df_sample = training_data_df.sample(frac = config.TFIDF_SAMPLE_FRAC, random_state = 42)
    training_data_df_sample.reset_index(inplace = True, drop = True)

    # Get human and synthetic text
    text_samples = {
        'human': training_data_df_sample['String'][training_data_df_sample['Source'] == 'human'],
        'synthetic': training_data_df_sample['String'][training_data_df_sample['Source'] == 'synthetic']
    }

    # Loop on the training & testing datsets
    for text_source, text in text_samples.items():

        # Split the data up for n workers
        texts_chunks = np.array_split(text, config.KL_SCORE_WORKERS)

        # Start multiprocessing manager and use it to create and empty list
        # in shared memory to get data back from workers
        manager = multiprocessing.Manager()
        return_list = manager.list()
        
        # Loop on text chunks, submitting each for processing
        jobs = []

        for texts_chunk in texts_chunks:
            p = multiprocessing.Process(target = helper_funcs.submitt_text_for_cleaning, args=(texts_chunk, return_list))
            jobs.append(p)
            p.start()

        # Call join on each worker in the jobs list
        for proc in jobs:
            proc.join()

        # Get the cleaned text back from shared memory and place it in the text samples dict
        text_samples[text_source] = return_list


    # Calculate TF-IDF on the cleaned samples, format the results as a dict.
    # of look-up tables
    tfidf_luts = {}

    # Loop on the training & testing datsets
    for text_source, text in text_samples.items():

        # Start multiprocessing manager and use it to create and empty dict
        # in shared memory to get data back from worker
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # Run the TF-IDF vectorization in a worker process so we can
        # be sure we get the memory back when the process exits
        p = multiprocessing.Process(target = helper_funcs.make_tfidf_lut, args = (text, return_dict))

        # Run the job
        p.start()
        p.join()

        tfidf_luts['text_source'] = return_dict


    # Loop on the training & testing datsets
    texts = {
        'human': training_data_df['String'][training_data_df['Source'] == 'human'],
        'synthetic': training_data_df['String'][training_data_df['Source'] == 'synthetic']
    }

    # Empty holder for results
    fragment_scores = {}

    for text_source, text in texts.items():

        # Split the data up for n workers
        texts_chunks = np.array_split(text, config.KL_SCORE_WORKERS)

        # Start multiprocessing manager and use it to create and empty list
        # in shared memory to get data back from workers
        manager = multiprocessing.Manager()
        return_list = manager.list()
        
        # Loop on text chunks, submitting each for processing
        jobs = []

        for texts_chunk in texts_chunks:
            p = multiprocessing.Process(target = helper_funcs.make_tfidf_lut, args=(texts_chunk, return_dict))
            jobs.append(p)
            p.start()

        # Call join on each worker in the jobs list
        for proc in jobs:
            proc.join()

        # Get the TF-IDF lut dict back from shared memory
        fragment_scores[text_source] = return_dict


    # Get a kernel density estimate of the KL divergence so we can use the
    # associated probability density function to convert perplexity ratio scores
    # into KL scores

    # Get a list of points covering the range of score values
    tfidf_scores = fragment_scores['human'] + fragment_scores['synthetic']
    x = np.arange(min(tfidf_scores) - 2, max(tfidf_scores) + 2, 0.01).tolist()

    # Do the exponential gaussian fits and get values for f(x)
    bounds = [[-10.0,10.0],[-10.0,10.0],[-10.0,10.0]]

    human_exponnorm = fit(exponnorm, fragment_scores['human'], bounds = bounds)
    human_exponnorm_fit = exponnorm(human_exponnorm.params.K, human_exponnorm.params.loc, human_exponnorm.params.scale).pdf(x)

    synthetic_exponnorm = fit(exponnorm, fragment_scores['synthetic'], bounds = bounds)
    synthetic_exponnorm_fit = exponnorm(synthetic_exponnorm.params.K, synthetic_exponnorm.params.loc, synthetic_exponnorm.params.scale).pdf(x)

    # Calculate the KL divergence of the fitted values
    kl = helper_funcs.kl_divergence(synthetic_exponnorm_fit, human_exponnorm_fit)

    # Convert the kl 'density' values into integer 'count' values
    kl = kl + abs(min(kl))
    kl = kl * 100
    kl_counts = [int(density) for density in kl]

    # Now, construct a list where each value of x appears a number of times
    # equal to it's kl 'count'
    kl_scores = []

    for i in range(len(kl_counts)):
        kl_scores.extend([x[i]] * kl_counts[i])

    # Finally, run a KDE on the reconstructed KL scores
    kl_kde = gaussian_kde(kl_scores)

    return kl_kde