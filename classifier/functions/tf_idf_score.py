'''Collection of functions to calculate product normalized TF-IDF score
on binned data. Parallelizes calculation over bins.'''

from __future__ import annotations
from typing import Callable

import re
import gc
import h5py
import nltk
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import functions.multiprocess_logging as log_funcs
import configuration as config


# Get and set up stop words and an instance of the Word Net
# Lemmatizer for use in cleaning text for vectorization
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def tf_idf_score(
        hdf5_file: str,
        score_sample: bool = False,
        logfile_name: str = 'tf_idf.log'
) -> None:

    '''Main function to parallelize computation of TF-IDF score
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

        # Pull the testing features for this bin
        bin_testing_features_df = data_lake[f'testing/{bin_id}/features']

        # Take sample if desired
        if score_sample is True:
            bin_training_features_df = bin_training_features_df.sample(frac = 0.1)
            bin_testing_features_df = bin_testing_features_df.sample(frac = 0.1)

        async_results.append(
            pool.apply_async(add_tf_idf_score,
                args = (
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


def add_tf_idf_score(
        bin_training_data_df: pd.DataFrame, 
        bin_testing_data_df: pd.DataFrame,
        worker_num: str,
        bin_id: int,
        logging_queue: Callable,
        configure_logging: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Takes training and testing datasets in dataframes. Uses training
    data to calculate term TF-IDF for human and synthetic data. Uses those
    term TF-IDF values to calculate product normalized TF-IDF score for
    each text fragment in the training and testing data. Adds TF-IDF score
    to dataframes as new features and return the updated dataframes.'''


    # Set-up logging
    configure_logging(logging_queue)
    logger = logging.getLogger(f'{__name__}.add_feature_kld_score')
    logger.info(f'Worker {worker_num} - {len(bin_training_data_df)} fragments in {bin_id}')

    try:
        human_texts, synthetic_texts = get_text(bin_training_data_df)

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_text() error: {err_string}')

    try:
        tfidf_luts = get_term_tf_idf(human_texts, synthetic_texts)

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - get_term_tf_idf() error: {err_string}')

    try:
        bin_training_data_df = tf_idf_score_text_fragments(bin_training_data_df, tfidf_luts)
        bin_testing_data_df = tf_idf_score_text_fragments(bin_testing_data_df, tfidf_luts)

    except Exception as err_string:
        logger.error(f'Worker {worker_num} - tf_idf_score_text_fragments() error: {err_string}')

    return bin_id, bin_training_data_df, bin_testing_data_df


def tf_idf_score_text_fragments(data_df: pd.DataFrame, tfidf_luts: dict = None) -> dict:
    '''Takes features dataframe and dictionary containing term TF-IDF look-up tables.
    scores text fragments from dataframe with product normalized difference in log2 TF-IDF mean.
    Adds TF-IDF score and log2 TF-IDF mean'''

    # Holders for new features
    tfidf_scores = []
    human_tfidf = []
    synthetic_tfidf = []

    # Get the text fragments
    texts = data_df['String']

    # Loop on dataframe rows
    for text in texts:

        # Clean the text
        text = clean_text(text)

        # Split the text into words
        words = text.split(' ')

        # Score the words using the human and synthetic luts
        human_tfidf_sum = 0
        synthetic_tfidf_sum = 0

        for word in words:

            if word in tfidf_luts['human'].keys():
                human_tfidf_sum += tfidf_luts['human'][word]

            if word in tfidf_luts['synthetic'].keys():
                synthetic_tfidf_sum += tfidf_luts['synthetic'][word]
            
        # Get the means
        human_tfidf_mean = human_tfidf_sum / len(words)
        synthetic_tfidf_mean = synthetic_tfidf_sum / len(words)

        # Get the product normalized TF-IDF score
        dmean_tfidf = human_tfidf_mean - synthetic_tfidf_mean
        product_normalized_dmean_tfidf = dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

        # Add to results
        human_tfidf.append(human_tfidf_mean)
        synthetic_tfidf.append(synthetic_tfidf_mean)
        tfidf_scores.append(product_normalized_dmean_tfidf)

    # Add new feature back to dataframe
    data_df['Human TF-IDF'] = human_tfidf
    data_df['Synthetic TF-IDF'] = synthetic_tfidf
    data_df['TF-IDF score'] = tfidf_scores

    return data_df


def get_term_tf_idf(human_texts: pd.Series, synthetic_texts: pd.Series) -> dict:
    '''Takes cleaned human and synthetic text as Pandas series, gets term TF-IDF values
    for each and returns as dictionary of look-up tables where key is term feature and
    value is term TF-IDF.'''

    # Dictionary to hold TF-IDF look-up tables
    tfidf_luts = {}

    # Loop twice to process the human and synthetic texts the same way
    for text_source, texts in zip(['human', 'synthetic'], [human_texts, synthetic_texts]):

        # Fit the TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectors = tfidf_vectorizer.fit_transform(texts)

        # Convert the vectors to numpy and replace zeros with NAN
        tfidf = tfidf_vectors.toarray()
        tfidf[tfidf == 0] = np.nan

        # Take the log2 and average the columns (i.e. get average TF-IDF per word)
        log_tfidf = np.log2(tfidf)
        log_tfidf_mean = np.nanmean(log_tfidf, axis = 0)

        # Get the words
        features = tfidf_vectorizer.get_feature_names_out()

        # Release some memory
        del tfidf_vectorizer
        del tfidf_vectors
        _ = gc.collect()

        # Add result to look-up table
        tfidf_luts[text_source] = dict(zip(features, log_tfidf_mean))

    return tfidf_luts


def get_text(bin_training_data_df: pd.DataFrame) ->tuple[pd.Series, pd.Series]:
    '''Gets and cleans human and synthetic text from training data.'''

    # If we have more than 10,000 text fragments, take a random sample of 10,000
    # to keep memory utilization under control during vectorization, then get the
    # text strings for the human and synthetic text fragments in the sample
    if len(bin_training_data_df) > 10000:
    
        training_data_df_sample = bin_training_data_df.sample(n = 10000, random_state = 42)
        training_data_df_sample.reset_index(inplace = True, drop = True)

        human_texts = training_data_df_sample['String'][training_data_df_sample['Source'] == 'human']
        synthetic_texts = training_data_df_sample['String'][training_data_df_sample['Source'] == 'synthetic']

    # If the dataset has 10,000 or less text fragments, directly pull all of the
    # text fragment strings for human and synthetic fragments from the data
    else:
        
        human_texts = bin_training_data_df['String'][bin_training_data_df['Source'] == 'human']
        synthetic_texts = bin_training_data_df['String'][bin_training_data_df['Source'] == 'synthetic']

    # Clean text for vectorization
    human_texts = human_texts.apply(lambda x: clean_text(x))
    synthetic_texts = synthetic_texts.apply(lambda x: clean_text(x))

    return human_texts, synthetic_texts


# Get and set up stop words and an instance of the Word Net
# Lemmatizer for use in cleaning text for vectorization
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def clean_text(text: str = None) -> str:
    '''Takes a text string and cleans it for vectorization.
    Returns cleaned text as string.'''
    
    # Lowercase everything
    text = text.lower()

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # Remove URLs 
    text = re.sub(r"http\S+", "",text)
    
    # Remove html tags
    html = re.compile(r'<.*?>') 
    text = html.sub(r'',text)
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

    # Remove punctuations
    for p in punctuations:
        text = text.replace(p,'')
        
    # Remove stopwords
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    return text