'''Project specific constants and configuration. Non-HuggingFace 
defaults for various model parameters'''

import os
import torch

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

# Set mode to testing to mock scoring function with random output
# between 0.0 and 1.0 and not load any LLMs. Set to production
# to run real scoring function.
MODE = 'production'

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH=os.path.dirname(os.path.realpath(__file__))
MODULE_PARENT_PATH = os.path.abspath(os.path.join(PROJECT_ROOT_PATH, os.pardir))
TELEGRAM_BOT_PATH = f'{MODULE_PARENT_PATH}/telegram_bot'

# Other project paths
LOG_PATH=f'{PROJECT_ROOT_PATH}/logs'
DATA_PATH=f'{PROJECT_ROOT_PATH}/data'
FRAGMENT_TURNAROUND_DATA = f'{TELEGRAM_BOT_PATH}/logs/fragment_turnaround.dat'

# Logging stuff
PLOT_BOT_TRAFFIC = False
LOG_LEVEL='INFO'
LOG_PREFIX='%(levelname)s - %(message)s'
CLEAR_LOGS=True

# Flask app stuff
HOST_IP=os.environ['HOST_IP']
FLASK_PORT=os.environ['FLASK_PORT']
REDIS_IP=os.environ['REDIS_IP']
REDIS_PORT=os.environ['REDIS_PORT']
REDIS_PASSWORD=os.environ['REDIS_PASSWORD']
REDIS_URL=f'redis://:{REDIS_PASSWORD}@{REDIS_IP}:{REDIS_PORT}'

# Models to use for scoring
READER_MODEL='meta-llama/Meta-Llama-3-8B'
READER_DEVICE='cuda:1'

WRITER_MODEL='meta-llama/Meta-Llama-3-8B-instruct'
WRITER_DEVICE='cuda:2'

CALCULATION_DEVICE='cuda:2'

PERPLEXITY_RATIO_KLD_KDE = f'{DATA_PATH}/perplexity_ratio_KLD_KDE.pkl'
TFIDF_LUT = f'{DATA_PATH}/TFIDF_lut.pkl'
TFIDF_SCORE_KLD_KDE = f'{DATA_PATH}/TFIDF_score_KLD_KDE.pkl'
XGBOOST_CLASSIFIER = f'{DATA_PATH}/XGBoost_classifier.pkl'

######################################################################
# NON-HF default model parameters ####################################
######################################################################

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
