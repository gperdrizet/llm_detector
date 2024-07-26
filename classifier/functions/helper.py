'''Helper functions for running jobs'''

import re
import pathlib
import numpy as np
import pandas as pd
import nltk
from math import log2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import classifier.configuration as config

def force_after(task_name: str = None):
    '''Forces all to be re-run starting with given task by removing their output'''

    # Dictionary of string task names and their output files
    tasks = {
        'LoadData': config.LOADED_DATA,
        'PerplexityRatioKLD': config.PERPLEXITY_RATIO_KLD_KDE,
        'AddPerplexityRatioKLDScore': config.PERPLEXITY_RATIO_KLD_SCORE_ADDED,
        'TFIDFScoreKLD': config.TFIDF_SCORE_KLD_KDE
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


def add_kl_divergence_score(data_chunk: pd.DataFrame = None, kl_kde = None, return_list = None):
    '''Calculates and adds perplexity ratio Kulback-Leibler divergence to a
    dataframe chunk. Added result to shared memory list.'''

    kl_scores = kl_kde.pdf(data_chunk['Perplexity ratio score'])
    data_chunk['Perplexity ratio Kullback-Leibler score'] = kl_scores

    return_list.append(data_chunk)


def kl_divergence(p: list = None, q: list = None) -> list:
    '''Takes two lists, calculates KL divergence'''

    return [p[i] * log2(p[i]/q[i]) for i in range(len(p))]


def clean_ooms(dataframe: pd.DataFrame = None) -> pd.DataFrame:
    '''Removes string NAN and OOM error placeholders.'''

    # Replace and remove string 'OOM' and 'NAN' values
    dataframe.replace('NAN', np.nan, inplace = True)
    dataframe.replace('OOM', np.nan, inplace = True)
    dataframe.dropna(inplace = True)

    return dataframe


def fix_dtypes(dataframe: pd.DataFrame = None) -> pd.DataFrame:
    '''Enforces correct dtypes on feature columns'''

    # Set dtypes
    dataframe = dataframe.astype({
        'Fragment length (tokens)': int, 
        'Perplexity': float,
        'Cross-perplexity': float,
        'Perplexity ratio score': float
    })

    return dataframe

def submitt_text_for_cleaning(texts_chunk: list = None, return_list = None):
    '''Submits chunk of texts for cleaning, adds results to shared memory list.'''

    nltk.download('stopwords')
    nltk.download('wordnet')
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    cleaned_texts = []

    for text in texts_chunk:

        cleaned_texts.append(
            clean_text(
                text = text,
                sw = sw,
                lemmatizer = lemmatizer
            )
        )

    return_list.extend(cleaned_texts)
    

def clean_text(text: str = None, sw = None, lemmatizer = None) -> str:
    '''Cleans up text string for TF-IDF'''
    
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


def make_tfidf_lut(texts: list = None, return_dict = None) -> dict:
    '''Takes a list of text fragments, calculates TF-IDF and returns
    a dictionary look-up table with words as keys and TF-IDF value.'''

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

    return_dict = dict(zip(features, log_tfidf_mean))


def score_known_text_fragments(data_df: pd.DataFrame, tfidf_luts: dict = None) -> dict:
    '''Scores text fragments with product normalized difference in
    log2 TF-IDF mean.'''

    # Holders for TF-IDF values
    product_normalized_human_dmean_tfidf = []
    product_normalized_synthetic_dmean_tfidf = []

    # Loop on dataframe rows
    for _, row in data_df.iterrows():
        
        human_tfidf_sum = 0
        synthetic_tfidf_sum = 0

        # Get the text from this row
        text = row['String']

        # Clean the text
        text = clean_text(text)

        # Split the text into words
        words = text.split(' ')

        # Score the words using the human and synthetic luts
        for word in words:

            if word in tfidf_luts['human'].keys():
                human_tfidf_sum += tfidf_luts['human'][word]

            if word in tfidf_luts['synthetic'].keys():
                synthetic_tfidf_sum += tfidf_luts['synthetic'][word]

        # Get the means
        human_tfidf_mean = human_tfidf_sum / len(words)
        synthetic_tfidf_mean = synthetic_tfidf_sum / len(words)
        dmean_tfidf = human_tfidf_mean - synthetic_tfidf_mean
        product_normalized_dmean_tfidf = dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

        if row['Source'] == 'human':
            product_normalized_human_dmean_tfidf.append(product_normalized_dmean_tfidf)

        elif row['Source'] == 'synthetic':
            product_normalized_synthetic_dmean_tfidf.append(product_normalized_dmean_tfidf)

    return {'human': product_normalized_human_dmean_tfidf, 'synthetic': product_normalized_synthetic_dmean_tfidf}