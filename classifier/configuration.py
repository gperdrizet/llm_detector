'''Project specific constants and configuration.'''

import os
import multiprocessing
import torch

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Logging
LOG_LEVEL = 'DEBUG'
LOG_PATH = f'{PROJECT_ROOT_PATH}/logs'
LOG_PREFIX = '%(levelname)s - %(name)s - %(message)s'

# Other project paths
DATA_PATH = f'{PROJECT_ROOT_PATH}/data'
BENCHMARKING_DATA_PATH = f'{DATA_PATH}/benchmarking'
HANS_DATA_PATH = f'{DATA_PATH}/hans_2024'

# Data paths
RAW_INPUT_DATA = f'{HANS_DATA_PATH}/falcon-7b_scores_v2_10-300_words.json'
LENGTH_BINNED_DATASET = f'{DATA_PATH}/length_binned_dataset.h5'

# Luigi feature engineering/data pipeline parameters
MODELS_PATH = f'{DATA_PATH}/models'
PERPLEXITY_RATIO_KL_KDE = f'{MODELS_PATH}/perplexity_ratio_KL_KDE.pkl'
TFIDF_KL_KDE = f'{MODELS_PATH}/TFIDF_KL_KDE.pkl'
XGB_CLASSIFIER = f'{MODELS_PATH}/XGB_classifier.pkl'

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

# Perplexity ratio stuff
READER_MODEL='tiiuae/falcon-7b'
READER_DEVICE='cuda:1'

WRITER_MODEL='tiiuae/falcon-7b-instruct'
WRITER_DEVICE='cuda:2'

CALCULATION_DEVICE='cuda:0'

# Loading details
DEFAULT_CACHE_DIR='/mnt/fast_scratch/huggingface_transformers_cache'
DEFAULT_HF_MODEL_STRING='meta-llama/Meta-Llama-3-8B'
DEFAULT_MODEL_NAME='LLaMA3'
DEFAULT_DEVICE_MAP='cuda:0'
DEFAULT_CPU_CORES=16
DEFAULT_AVAILABLE_GPUS=['cuda:0', 'cuda:1', 'cuda:2']

# Quantization configuration defaults
DEFAULT_QUANTIZATION='4-bit'
DEFAULT_BNB_4BIT_COMPUTE_DTYPE=torch.float16

# Generation configuration defaults
DEFAULT_MAX_NEW_TOKENS=32

# Decoding strategy
DEFAULT_DECODING_STRATEGY=None


# Older stuff for earlier iterations of the feature engineering pipeline
FEATURE_ENGINEERING_CLASS_INSTANCE = f'{HANS_DATA_PATH}/feature_engineering_class_instance.plk'

COMBINED_SCORED_HANS_DATA = f'{HANS_DATA_PATH}/all-scores_falcon_v2_no_PMC.json'
COMBINED_SCORED_HANS_TRAINING_DATA = f'{HANS_DATA_PATH}/training-scores_falcon_v2_no_PMC.json'
COMBINED_SCORED_HANS_TESTING_DATA = f'{HANS_DATA_PATH}/testing-scores_falcon_v2.json'
COMBINED_SCORED_HANS_TRAINING_DATA_PR = f'{HANS_DATA_PATH}/training-scores-PR_falcon_v2_no_PMC.json'
COMBINED_SCORED_HANS_TESTING_DATA_PR = f'{HANS_DATA_PATH}/testing-scores-PR_falcon_v2_no_PMC.json'
COMBINED_SCORED_HANS_TRAINING_DATA_PR_TFIDF = f'{HANS_DATA_PATH}/training-scores-PR-TFIDF_falcon_v2_no_PMC.json'
COMBINED_SCORED_HANS_TESTING_DATA_PR_TFIDF = f'{HANS_DATA_PATH}/testing-scores-PR-TFIDF_falcon_v2_no_PMC.json'

# Legacy data paths for experimentation notebooks
COMBINED_SCORED_HANS_DATA_PR_TFIDF = f'{HANS_DATA_PATH}/all-scores-PR-TFIDF.json'