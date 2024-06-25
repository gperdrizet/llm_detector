'''Project specific constants and configuration. Non-HuggingFace 
defaults for various model parameters'''

import os
import torch

######################################################################
# Project meta-stuff: paths, logging, etc. ###########################
######################################################################

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

######################################################################
# Default model parameters ###########################################
######################################################################

# Loading details
CACHE_DIR='/mnt/fast_scratch/huggingface_transformers_cache'
HF_MODEL_STRING='meta-llama/Meta-Llama-3-8B'
MODEL_NAME='LLaMA3'
DEVICE_MAP='cuda:0'
CPU_CORES=16
AVAILABLE_GPUS=['cuda:0', 'cuda:1', 'cuda:2']

# Quantization configuration defaults
QUANTIZATION='4-bit'
BNB_4BIT_COMPUTE_DTYPE=torch.float16

# Generation configuration defaults
MAX_NEW_TOKENS=32

# Decoding strategy
DECODING_STRATEGY=None
