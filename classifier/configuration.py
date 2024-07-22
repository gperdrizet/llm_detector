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
COMBINED_SCORED_HANS_DATA_KL = f'{HANS_DATA_PATH}/all-scores-KL.json'