'''Project specific constants and configuration. Non-HuggingFace 
defaults for various model parameters'''

import os
import torch

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

# Set mode to testing to mock scoring function with random output
# between 0.0 and 1.0 and not load any LLMs. Set to production
# to run real scoring function
MODE = 'testing'

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH=os.path.dirname(os.path.realpath(__file__))

# Other project paths
LOG_PATH=f'{PROJECT_ROOT_PATH}/logs'

# Logging stuff
LOG_LEVEL='DEBUG'
LOG_PREFIX='%(levelname)s - %(message)s'
CLEAR_LOGS=True

# Flask app stuff
IP_ADDRESS='192.168.1.148'
PORT=5000
REDIS_URL='redis://192.168.1.148'

# Models to use for scoring
OBSERVER_MODEL='meta-llama/Meta-Llama-3-8B'
OBSERVER_DEVICE='cuda:1'

PERFORMER_MODEL='meta-llama/Meta-Llama-3-8B-instruct'
PERFORMER_DEVICE='cuda:2'

CALCULATION_DEVICE='cuda:0'

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
