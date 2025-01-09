'''Helper functions for running jobs'''

from __future__ import annotations
from typing import Callable

import re
import pathlib
import logging
from math import log2

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

import configuration as config

# def start_logger(
#         logfile_name: str='llm_detector.log',
#         logger_name: str='benchmarking'
# ) -> Callable:

#     '''Sets up logging, returns logger'''

#     # Build logfile name
#     logfile = f'{config.LOG_PATH}/{logfile_name}'
#     print(f'Will log to: {logfile}\n')

#     # Clear logs from previous runs
#     pathlib.Path(logfile).unlink(missing_ok=True)

#     # Create logger
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(config.LOG_LEVEL)

#     handler = logging.handlers.RotatingFileHandler(
#         f'{config.LOG_PATH}/{logfile_name}',
#         encoding = 'utf-8',
#         maxBytes = 1 * 1024 * 1024,  # 1 MiB
#         backupCount = 5
#     )

#     formatter = logging.Formatter(config.LOG_PREFIX,
#                                   datefmt = '%Y-%m-%d %I:%M:%S %p')

#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

#     return logger


def force_after(task_name: str = None):
    '''Forces all to be re-run starting with given task by removing their output'''

    # Dictionary of string task names and their output files
    tasks = {
        'LoadData': config.LOADED_DATA,
        'PerplexityRatioKLD': config.PERPLEXITY_RATIO_KLD_KDE,
        'AddPerplexityRatioKLDScore': config.PERPLEXITY_RATIO_KLD_SCORE_ADDED,
        'MakeTFIDFLut': config.TFIDF_LUT,
        'AddTFIDFScore': config.TFIDF_SCORE_ADDED,
        'TFIDFScoreKLD': config.TFIDF_SCORE_KLD_KDE,
        'AddTFIDFKLDScore': config.TFIDF_KLD_SCORE_ADDED,
        'TrainXGBoost': config.XGBOOST_CLASSIFIER
    }

    # Loop on the task dictionary
    remove_output = False

    for task, output_file in tasks.items():

        # When we find the task, flip the value of remove_output to True
        # so that we will remove the output files for this and all
        # subsequent tasks
        if task == task_name:
            remove_output = True

        # If the flag has been flipped remove the output file
        if remove_output is True:
            pathlib.Path(output_file).unlink(missing_ok = True)


# def add_perplexity_ratio_kl_divergence_score(
#         data_chunk: pd.DataFrame = None,
#         kl_kde = None,
#         return_list = None
# ):
#     '''Calculates and adds perplexity ratio Kullback-Leibler divergence to a
#     dataframe chunk. Added result to shared memory list.'''

#     kl_scores = kl_kde.pdf(data_chunk['Perplexity ratio score'])
#     data_chunk['Perplexity ratio Kullback-Leibler score'] = kl_scores

#     return_list.append(data_chunk)

# def add_tfidf_kl_divergence_score(
#         data_chunk: pd.DataFrame = None,
#         kl_kde = None,
#         return_list = None
# ):
#     '''Calculates and adds tfidf Kullback-Leibler divergence to a
#     dataframe chunk. Added result to shared memory list.'''

#     kl_scores = kl_kde.pdf(data_chunk['TF-IDF score'])
#     data_chunk['TF-IDF Kullback-Leibler score'] = kl_scores

#     return_list.append(data_chunk)


def clean_ooms(dataframe: pd.DataFrame = None) -> pd.DataFrame:
    '''Removes string NAN and OOM error placeholders.'''

    # Replace and remove string 'OOM' and 'NAN' values
    dataframe.replace('NAN', np.nan, inplace = True)
    dataframe.replace('OOM', np.nan, inplace = True)
    dataframe.dropna(inplace = True)

    return dataframe


# def fix_dtypes(dataframe: pd.DataFrame = None) -> pd.DataFrame:
#     '''Enforces correct dtypes on feature columns'''

#     # Set dtypes
#     dataframe = dataframe.astype({
#         'Fragment length (tokens)': int, 
#         'Perplexity': float,
#         'Cross-perplexity': float,
#         'Perplexity ratio score': float
#     })

#     return dataframe

# def submitt_text_for_cleaning(texts_chunk: list = None, return_list = None):
#     '''Submits chunk of texts for cleaning, adds results to shared memory list.'''

#     sw = stopwords.words('english')
#     lemmatizer = WordNetLemmatizer()

#     cleaned_texts = []

#     for text in texts_chunk:

#         cleaned_texts.append(
#             clean_text(
#                 text = text,
#                 sw = sw,
#                 lemmatizer = lemmatizer
#             )
#         )

#     return_list.extend(cleaned_texts)


# def clean_text(text: str = None, sw = None, lemmatizer = None) -> str:
#     '''Cleans up text string for TF-IDF'''

#     # Lowercase everything
#     text = text.lower()

#     # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
#     text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

#     # Remove URLs
#     text = re.sub(r"http\S+", "",text)

#     # Remove html tags
#     html = re.compile(r'<.*?>')
#     text = html.sub(r'',text)

#     punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

#     # Remove punctuations
#     for p in punctuations:
#         text = text.replace(p,'')

#     # Remove stopwords
#     text = [word.lower() for word in text.split() if word.lower() not in sw]
#     text = [lemmatizer.lemmatize(word) for word in text]
#     text = " ".join(text)

#     # Remove emojis
#     emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#     "]+", flags=re.UNICODE)

#     text = emoji_pattern.sub(r'', text)

#     return text


# def make_tfidf_lut(texts: list = None, text_source: str = None, return_dict = None) -> dict:
#     '''Takes a list of text fragments, calculates TF-IDF and returns
#     a dictionary look-up table with words as keys and TF-IDF value.'''

#     # Fit the TF-IDF vectorizer
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_vectors = tfidf_vectorizer.fit_transform(texts)

#     # Convert the vectors to numpy and replace zeros with NAN
#     tfidf = tfidf_vectors.toarray()
#     tfidf[tfidf == 0] = np.nan

#     # Take the log2 and average the columns (i.e. get average TF-IDF per word)
#     log_tfidf = np.log2(tfidf)
#     log_tfidf_mean = np.nanmean(log_tfidf, axis = 0)

#     # Get the words
#     features = tfidf_vectorizer.get_feature_names_out()

#     return_dict[text_source] = dict(zip(features, log_tfidf_mean))


# def tfidf_score_text_fragments(
#         data_chunk: pd.DataFrame,
#         tfidf_luts: dict = None,
#         return_list = None
# ) -> dict:
#     '''Scores text fragments with product normalized difference in
#     log2 TF-IDF mean.'''

#     # Holders for TF-IDF values
#     human_tfidf_means = []
#     synthetic_tfidf_means = []
#     product_normalized_dmean_tfidfs = []

#     # Stop words and lemmatizer for text cleaning
#     sw = stopwords.words('english')
#     lemmatizer = WordNetLemmatizer()

#     # Loop on dataframe rows
#     for _, row in data_chunk.iterrows():

#         human_tfidf_sum = 0
#         synthetic_tfidf_sum = 0

#         # Get the text from this row
#         text = row['String']

#         # Clean the text
#         text = clean_text(
#             text = text,
#             sw = sw,
#             lemmatizer = lemmatizer
#         )

#         # Split the text into words
#         words = text.split(' ')

#         # Score the words using the human and synthetic luts
#         for word in words:

#             if word in tfidf_luts['human'].keys():
#                 human_tfidf_sum += tfidf_luts['human'][word]

#             if word in tfidf_luts['synthetic'].keys():
#                 synthetic_tfidf_sum += tfidf_luts['synthetic'][word]

#         # Get the means
#         human_tfidf_mean = human_tfidf_sum / len(words)
#         synthetic_tfidf_mean = synthetic_tfidf_sum / len(words)
#         dmean_tfidf = human_tfidf_mean - synthetic_tfidf_mean
#         product_normalized_dmean_tfidf = dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

#         human_tfidf_means.append(human_tfidf_mean)
#         synthetic_tfidf_means.append(synthetic_tfidf_mean)
#         product_normalized_dmean_tfidfs.append(product_normalized_dmean_tfidf)

#     data_chunk['Human TF-IDF'] = human_tfidf_means
#     data_chunk['Synthetic TF-IDF'] = synthetic_tfidf_means
#     data_chunk['TF-IDF score'] = product_normalized_dmean_tfidfs

#     return_list.append(data_chunk)


# def make_padded_range(data: np.array, n_points: int = 100) -> np.array:
#     '''Takes an input array and optionally a number of points. Generates
#     a set of n_points sample points which span the data's range plus
#     10% on either end. Returns the sample points.'''

#     # Find the range of the data
#     data_range = max(data) - min(data)

#     # Calculate the padding amount as 10% of the data's range
#     padding = data_range * 0.1

#     # Get the padded min and max values
#     x_max = max(data) + padding
#     x_min = min(data) - padding

#     # Determine the sampling frequency based on the caller
#     # specified number of points
#     sample_frequency = (x_max - x_min) / n_points

#     # Make the points
#     x = np.arange(x_min, x_max, sample_frequency)

#     return x

# def sigma_clip_data(data: np.array, n_sigma: float = 5.0) -> np.array:
#     '''Takes data and removes any points above or below the
#     specified number of standard deviations away from the mean.'''

#     # Get mean and standard deviation
#     data_std_dev = np.std(data)
#     data_mean = np.mean(data)

#     # Remove points that more than n standard deviations below mean
#     mask = data > data_mean - (data_std_dev * n_sigma)
#     data = data[mask]

#     # Remove data that are more than n standard deviations above the mean
#     mask = data < data_mean + (data_std_dev * n_sigma)
#     data = data[mask]

#     return data