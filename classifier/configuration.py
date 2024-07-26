'''Project specific constants and configuration.'''

import os

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

# Feature engineering/data pipeline parameters
SCORED_HANS_DATASETS = {
    'cc_news': 'cc_news-scores.json',
    'pubmed': 'pubmed-scores.json',
    'cnn': 'cnn-scores.json'
}

TRAIN_TEST_SPLIT = 0.8

# Luigi data pipeline filepaths
LUIGI_DATA = f'{DATA_PATH}/luigi'
LOADED_DATA = f'{LUIGI_DATA}/01-loaded.json'