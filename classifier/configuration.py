'''Project specific constants and configuration.'''

import os
import multiprocessing

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Other project paths
DATA_PATH = f'{PROJECT_ROOT_PATH}/data'
BENCHMARKING_DATA_PATH = f'{DATA_PATH}/benchmarking'
HANS_DATA_PATH = f'{DATA_PATH}/hans_2024'
COMBINED_SCORED_HANS_DATA = f'{HANS_DATA_PATH}/all-scores.json'
COMBINED_SCORED_HANS_TRAINING_DATA = f'{HANS_DATA_PATH}/training-scores.json'
COMBINED_SCORED_HANS_TESTING_DATA = f'{HANS_DATA_PATH}/testing-scores.json'
COMBINED_SCORED_HANS_TRAINING_DATA_PR = f'{HANS_DATA_PATH}/training-scores-PR.json'
COMBINED_SCORED_HANS_TESTING_DATA_PR = f'{HANS_DATA_PATH}/testing-scores-PR.json'
COMBINED_SCORED_HANS_TRAINING_DATA_PR_TFIDF = f'{HANS_DATA_PATH}/training-scores-PR-TFIDF.json'
COMBINED_SCORED_HANS_TESTING_DATA_PR_TFIDF = f'{HANS_DATA_PATH}/testing-scores-PR-TFIDF.json'

MODELS_PATH = f'{DATA_PATH}/models'
PERPLEXITY_RATIO_KL_KDE = f'{MODELS_PATH}/perplexity_ratio_KL_KDE.pkl'
TFIDF_KL_KDE = f'{MODELS_PATH}/TFIDF_KL_KDE.pkl'
XGB_CLASSIFIER = f'{MODELS_PATH}/XGB_classifier.pkl'

# Luigi feature engineering/data pipeline parameters
KL_SCORE_WORKERS = multiprocessing.cpu_count() - 2

SCORED_HANS_DATASETS = {
    'cc_news': 'cc_news-scores.json',
    'pubmed': 'pubmed-scores.json',
    'cnn': 'cnn-scores.json'
}

TRAIN_TEST_SPLIT = 0.8
TFIDF_SAMPLE_FRAC = 0.5

# Luigi pipeline filepaths
LUIGI_DATA_PATH = f'{DATA_PATH}/luigi'
LOADED_DATA = f'{LUIGI_DATA_PATH}/01-loaded.json'
PERPLEXITY_RATIO_KLD_SCORE_ADDED = f'{LUIGI_DATA_PATH}/02-perplexity_ratio_KLD_score_added.json'
TFIDF_SCORE_ADDED = f'{LUIGI_DATA_PATH}/03-TFIDF_score_added.json'
TFIDF_KLD_SCORE_ADDED = f'{LUIGI_DATA_PATH}/04-TFIDF_KLD_score_added.json'

# Luigi pipeline models
MODELS_PATH = f'{DATA_PATH}/models'
PERPLEXITY_RATIO_KLD_KDE = f'{MODELS_PATH}/perplexity_ratio_KLD_KDE.pkl'
TFIDF_LUT =  f'{MODELS_PATH}/TFIDF_lut.pkl'
TFIDF_SCORE_KLD_KDE = f'{MODELS_PATH}/TFIDF_score_KLD_KDE.pkl'
XGBOOST_CLASSIFIER = f'{MODELS_PATH}/XGBoost_classifier.pkl'
