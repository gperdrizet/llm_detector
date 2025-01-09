'''Collection of top level functions for Luigi tasks in the 
data & feature engineering pipeline'''

from __future__ import annotations

import gc
import pickle
import json
import multiprocessing
import nltk
import xgboost
#from math import log2

import numpy as np
import pandas as pd
from scipy.stats import fit, exponnorm, gaussian_kde
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

import functions.helper as helper_funcs
import configuration as config

nltk.download('stopwords')
nltk.download('wordnet')

def load_data() -> dict:
    '''Parses and combines perplexity ratio scored text fragments from
    all three Hans 2024 datasets. Returns results as training testing
    dict of JSON.'''

    # Holder for each parsed dataset
    dataframes=[]

    for _, filename in config.SCORED_HANS_DATASETS.items():

        # Load the data
        dataframe=pd.read_json(f'{config.HANS_DATA_PATH}/{filename}')

        # Update name of some columns, some scoring runs used the earlier names
        dataframe.rename(columns={
            'Binoculars score': 'Perplexity ratio score',
            'Observer peak memory (GB)': 'Reader peak memory (GB)',
            'Performer peak memory (GB)': 'Writer peak memory (GB)'
        }, inplace=True)

        # get rid of some unnecessary columns
        dataframe.drop([
            'Fragment', 
            'Reader peak memory (GB)', 
            'Writer peak memory (GB)'
        ], axis=1, inplace=True)

        # Replace and remove string 'OOM' and 'NAN' values
        dataframe=helper_funcs.clean_ooms(dataframe)

        # Fix some d-types
        dataframe=helper_funcs.fix_dtypes(dataframe)

        # Add this dataframe to the list
        dataframes.append(dataframe)

    # Combine the individual datasets and reset the index
    data_df=pd.concat(dataframes, axis=0)
    data_df.reset_index(inplace=True, drop=True)

    # Set length threshold
    data_df=data_df[data_df['Fragment length (tokens)'] > 50]

    # Split the data in to training and testing subsets.
    training_df=data_df.sample(frac=config.TRAIN_TEST_SPLIT, random_state=42)
    testing_df=data_df.drop(training_df.index)
    training_df.reset_index(inplace=True, drop=True)
    testing_df.reset_index(inplace=True, drop=True)

    # Construct a single dictionary containing the JSON of the training
    # and testing dataframes
    results={
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

    # Separate human and synthetic perplexity ratio scores
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

    # Get the Kullback-Leibler divergence of the fitted values
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


def add_perplexity_ratio_kld_score() -> dict:
    '''Uses multiprocessing to split text fragment data
    and add perpleixty ratio Kullback-Leibler divergence
    score to the chunks in parallel. Returns concatenated
    dataframes as training testing dict of JSON.'''

    # Load the data
    with open(config.LOADED_DATA, 'r', encoding='utf-8') as input_file:
        datasets=json.load(input_file)

    # Load the Kullback-Leibler divergence kernel density estimate
    with open(config.PERPLEXITY_RATIO_KLD_KDE, 'rb') as input_file:
        kl_kde=pickle.load(input_file)

    # Empty dict for results
    results={}

    # Loop on the training & testing datasets
    for dataset, data in datasets.items():

        data_df=pd.DataFrame.from_dict(json.loads(data))

        # Split the data up for n workers
        data_chunks=np.array_split(data_df, config.KL_SCORE_WORKERS)

        # Start multiprocessing manager and use it to create and empty list
        # in shard memory to get data back from workers
        manager=multiprocessing.Manager()
        return_list=manager.list()
        
        # Loop on data chunks, submitting each for processing
        jobs=[]

        for data_chunk in data_chunks:
            p=multiprocessing.Process(target=helper_funcs.add_perplexity_ratio_kl_divergence_score, args=(data_chunk, kl_kde, return_list))
            jobs.append(p)
            p.start()

        # Call join on each worker in the jobs list
        for proc in jobs:
            proc.join()

        # Concatenate the return chunks from the shared memory list into a single dataframe
        # and add it to the result as json
        result=pd.concat(return_list, axis=0)
        result.reset_index(inplace=True, drop=True)
        results[dataset] = result.to_json()

    return results


# def make_tfidf_lut() -> dict:
#     '''Calculates TF-IDF scores based on training data and creates
#     lookup tables for human and synthetic TFIDF values where key is work
#     and value is score.'''

#     # Load the data
#     with open(config.PERPLEXITY_RATIO_KLD_SCORE_ADDED, 'r', encoding='utf-8') as input_file:
#         data = json.load(input_file)

#     # Separate the training and testing data
#     data_dfs = {
#         'training': pd.DataFrame.from_dict(json.loads(data['training'])),
#         'testing': pd.DataFrame.from_dict(json.loads(data['testing']))
#     }

#     # Take samples so we dont' OOM during vectorization for TF-IDF
#     training_data_df_sample = data_dfs['training'].sample(frac = config.TFIDF_SAMPLE_FRAC, random_state = 42)
#     training_data_df_sample.reset_index(inplace = True, drop = True)

#     # Get human and synthetic text
#     text_samples = {
#         'human': training_data_df_sample['String'][training_data_df_sample['Source'] == 'human'],
#         'synthetic': training_data_df_sample['String'][training_data_df_sample['Source'] == 'synthetic']
#     }

#     # Loop on the training & testing datasets
#     for text_source, text in text_samples.items():

#         # Split the data up for n workers
#         texts_chunks = np.array_split(text, config.KL_SCORE_WORKERS)

#         # Start multiprocessing manager and use it to create and empty list
#         # in shared memory to get data back from workers
#         manager = multiprocessing.Manager()
#         return_list = manager.list()
        
#         # Loop on text chunks, submitting each for processing
#         jobs = []

#         for texts_chunk in texts_chunks:
#             p = multiprocessing.Process(target = helper_funcs.submitt_text_for_cleaning, args=(texts_chunk, return_list))
#             jobs.append(p)
#             p.start()

#         # Call join on each worker in the jobs list
#         for proc in jobs:
#             proc.join()

#         # Get the cleaned text back from shared memory and place it in the text samples dict
#         text_samples[text_source] = return_list


#     # Calculate TF-IDF on the cleaned samples, format the results as a dict.
#     # of look-up tables
#     tfidf_luts = {}

#     # Loop on the training & testing datasets
#     for text_source, text in text_samples.items():

#         # Start multiprocessing manager and use it to create and empty dict
#         # in shared memory to get data back from worker
#         manager = multiprocessing.Manager()
#         return_dict = manager.dict()

#         # Run the TF-IDF vectorization in a worker process so we can
#         # be sure we get the memory back when the process exits
#         p = multiprocessing.Process(target = helper_funcs.make_tfidf_lut, args = (text, text_source, return_dict))

#         # Run the job
#         p.start()
#         p.join()

#         tfidf_luts[text_source] = return_dict[text_source]

#     return tfidf_luts


# def add_tfidf_score() -> dict:
#     '''Uses TF-IDF lut from the previous step to calculate TF-IDF scores
#     for each text fragment in the training and testing data and add them
#     to the data.'''

#     # Load the luts
#     with open(config.TFIDF_LUT, 'rb') as input_file:
#         tfidf_luts = pickle.load(input_file)

#     # Load the data
#     with open(config.PERPLEXITY_RATIO_KLD_SCORE_ADDED, 'r', encoding='utf-8') as input_file:
#         data = json.load(input_file)

#     # Separate the training and testing data
#     data_dfs = {
#         'training': pd.DataFrame.from_dict(json.loads(data['training'])),
#         'testing': pd.DataFrame.from_dict(json.loads(data['testing']))
#     }

#     # Now use the luts to score all of the text

#     # Split the data up for n workers
#     for data_type, data_df in data_dfs.items():
#         texts_chunks = np.array_split(data_df, config.KL_SCORE_WORKERS)

#         # Start multiprocessing manager and use it to create and empty list
#         # in shared memory to get data back from workers
#         manager = multiprocessing.Manager()
#         return_list = manager.list()
        
#         # Loop on text chunks, submitting each for processing
#         jobs = []

#         for texts_chunk in texts_chunks:
#             p = multiprocessing.Process(target = helper_funcs.tfidf_score_text_fragments, args = (texts_chunk, tfidf_luts, return_list))
#             jobs.append(p)
#             p.start()

#         # Call join on each worker in the jobs list
#         for proc in jobs:
#             proc.join()

#         # Concatenate the return chunks from the shared memory list 
#         # back into the data dict
#         data_dfs[data_type] = pd.concat(return_list, axis = 0)
#         data_dfs[data_type].reset_index(inplace = True, drop = True)
#         data_dfs[data_type] = data_dfs[data_type].to_json()

#     return data_dfs

# def tfidf_score_kld_kde():
#     '''Calculates gaussian kernel density estimate of TF-IDF score.
#     Saves model for use in subsequent steps.'''

#     # Load the data
#     with open(config.TFIDF_SCORE_ADDED, 'r', encoding='utf-8') as input_file:
#         data = json.load(input_file)

#     training_data_df = pd.DataFrame.from_dict(json.loads(data['training']))

#     # Get a kernel density estimate of the KL divergence so we can use the
#     # associated probability density function to convert perplexity ratio scores
#     # into KL scores

#     # Get a list of points covering the range of score values
#     tfidf_scores = training_data_df['TF-IDF score']
#     x = np.arange(min(tfidf_scores) - 2, max(tfidf_scores) + 2, 0.01).tolist()

#     # Do the exponential gaussian fits and get values for f(x)
#     bounds = [[-10.0,10.0],[-10.0,10.0],[-10.0,10.0]]

#     human_scores = training_data_df['TF-IDF score'][training_data_df['Source'] == 'human']
#     human_exponnorm = fit(exponnorm, human_scores, bounds = bounds)
#     human_exponnorm_fit = exponnorm(human_exponnorm.params.K, human_exponnorm.params.loc, human_exponnorm.params.scale).pdf(x)

#     synthetic_scores = training_data_df['TF-IDF score'][training_data_df['Source'] == 'synthetic']
#     synthetic_exponnorm = fit(exponnorm, synthetic_scores, bounds = bounds)
#     synthetic_exponnorm_fit = exponnorm(synthetic_exponnorm.params.K, synthetic_exponnorm.params.loc, synthetic_exponnorm.params.scale).pdf(x)

#     # Calculate the KL divergence of the fitted values
#     kl = helper_funcs.kl_divergence(synthetic_exponnorm_fit, human_exponnorm_fit)

#     # Convert the kl 'density' values into integer 'count' values
#     kl = kl + abs(min(kl))
#     kl = kl * 100
#     kl_counts = [int(density) for density in kl]

#     # Now, construct a list where each value of x appears a number of times
#     # equal to it's kl 'count'
#     kl_scores = []

#     for i in range(len(kl_counts)):
#         kl_scores.extend([x[i]] * kl_counts[i])

#     # Finally, run a KDE on the reconstructed KL scores
#     kl_kde = gaussian_kde(kl_scores)

#     return kl_kde


# def add_tfidf_kld_score():
#     '''Uses multiprocessing to split text fragment data
#     and add TF-IDF Kullback-Leibler divergence
#     score to the chunks in parallel. Returns concatenated
#     dataframes as training testing dict of JSON.'''

#     # Load the data
#     with open(config.TFIDF_SCORE_ADDED, 'r', encoding = 'utf-8') as input_file:
#         datasets = json.load(input_file)

#     # Load the Kullback-Leibler divergence kernel density estimate
#     with open(config.TFIDF_SCORE_KLD_KDE, 'rb') as input_file:
#         kl_kde = pickle.load(input_file)

#     # Empty dict for results
#     results = {}

#     # Loop on the training & testing datasets
#     for dataset, data in datasets.items():

#         data_df = pd.DataFrame.from_dict(json.loads(data))

#         # Split the data up for n workers
#         data_chunks = np.array_split(data_df, config.KL_SCORE_WORKERS)

#         # Start multiprocessing manager and use it to create and empty list
#         # in shard memory to get data back from workers
#         manager = multiprocessing.Manager()
#         return_list = manager.list()
        
#         # Loop on data chunks, submitting each for processing
#         jobs = []

#         for data_chunk in data_chunks:
#             p = multiprocessing.Process(target = helper_funcs.add_tfidf_kl_divergence_score, args=(data_chunk, kl_kde, return_list))
#             jobs.append(p)
#             p.start()

#         # Call join on each worker in the jobs list
#         for proc in jobs:
#             proc.join()

#         # Concatenate the return chunks from the shared memory list into a single dataframe
#         # and add it to the result as json
#         result = pd.concat(return_list, axis = 0)
#         result.reset_index(inplace = True, drop = True)
#         results[dataset] = result.to_json()

#     return results


# def train_xgboost_classifier():
#     '''Trains and saves XGBoost classifier on completed data.'''

#     # Load the data
#     with open(config.TFIDF_KLD_SCORE_ADDED, 'r', encoding='utf-8') as input_file:
#         data = json.load(input_file)

#     # Separate the training and testing data
#     training_data_df = pd.DataFrame.from_dict(json.loads(data['training']))
#     testing_data_df = pd.DataFrame.from_dict(json.loads(data['testing']))

#     percent_human_fragments = (len(training_data_df[training_data_df['Source'] == 'human']) / len(training_data_df)) * 100
#     print(f'Text fragments are {percent_human_fragments}% human\n')

#     training_data_df.info()
#     print()
#     training_data_df.head()

#     # Split the data into features and labels
#     labels_train_df = training_data_df['Source']
#     features_train_df = training_data_df.drop('Source', axis = 1)
#     labels_test_df = testing_data_df['Source']
#     features_test_df = testing_data_df.drop('Source', axis = 1)

#     # Encode string class values as integers
#     label_encoder = LabelEncoder()
#     label_encoder = label_encoder.fit(labels_train_df)
#     labels_train = label_encoder.transform(labels_train_df)

#     label_encoder = LabelEncoder()
#     label_encoder = label_encoder.fit(labels_test_df)
#     labels_test = label_encoder.transform(labels_test_df)

#     # Keep dataframe copy for easy manipulation later and make a numpy copy for training
#     # without the dataset or string columns
#     features_train = features_train_df.drop(['Dataset', 'String'], axis = 1).to_numpy()
#     features_test = features_test_df.drop(['Dataset', 'String'], axis = 1).to_numpy()

#     print(f'Training data: {len(features_train)} examples')
#     print(f'Test data: {len(features_test)} examples')
#     print()
#     print('Training features:')
#     print(features_train_df.info())
#     print()
#     print('Training labels:')
#     print(labels_train)

#     # Fit model on training data
#     model = xgboost.XGBClassifier()
#     model.fit(features_train, labels_train)

#     # Make predictions for test data
#     y_pred = model.predict(features_test)
#     predictions = [round(value) for value in y_pred]

#     # Evaluate predictions
#     accuracy = accuracy_score(labels_test, predictions)
#     print('\nAccuracy: %.1f%%' % (accuracy * 100.0))

#     # Calculate confusion matrix
#     cm = confusion_matrix(labels_test, predictions)
#     print('\nConfusion matrix:')
#     print(cm)

#     # Normalize confusion matrix
#     print(f'\nNormalized confusion matrix:')
#     normalized_cm = cm / sum(sum(cm))
#     print(normalized_cm)
#     print()

#     return model