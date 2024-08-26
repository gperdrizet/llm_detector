'''Second generation functions for perplexity ratio scoring 
of data from https://arxiv.org/abs/2401.12070'''

from __future__ import annotations
from typing import Callable

import json
import random
import torch
import multiprocessing as mp

import benchmarking.configuration as config
import benchmarking.functions.helper as helper_funcs
import benchmarking.classes.llm as llm_class
from benchmarking.functions.metrics import perplexity, entropy

# Comment ##############################################################
# Code ########################################################################

def perplexity_ratio_score(output_file_name: str='perplexity_ratio_score_v2'):
    '''Main function to load and batch data and submit jobs.'''

    # Start logger
    logger = helper_funcs.start_logger('hans_data_perplexity_ratio_score_v2')
    logger.info('Starting v2 perplexity ratio score')

    # Input file
    input_datafile = f'{config.HANS_DATA_PATH}/aggregated_hans_data.jsonl'

    # Output file
    results_datafile = f'{config.HANS_DATA_PATH}/{output_file_name}'

    # Construct empty dictionary to collect results
    results = {}

    # Loop on variable names from config file
    for var_name in config.DEPENDENT_VARS:

        # Add and empty list for each variable to the results dictionary
        results[var_name] = []


    # Open the data so we can loop on the lines
    with open(input_datafile, encoding = 'utf-8') as f:

        # Make a set of batches of lines from the input data
        batches = []

        # Loop on worker count to make a batch for each one
        for i in range(config.WORKERS):

            # Make a batch
            batch = []

            # Loop on batch size
            for j in range(config.BATCH_SIZE):

                # Read next line from input file and add to batch
                batch.append(f.readline())

            # Once the batch is complete, add it to the batches
            batches.append(batch)

        logger.info(f'Have {len(batches)} batches of {len(batches[0])} lines for run.')

        # Instantiate pool with one worker per batch
        pool = mp.Pool(
            processes = len(batches),
            maxtasksperchild = 1
        )

        # Holder for returns from workers
        async_results = []

        # Loop on jobs for this run
        for i, batch in enumerate(batches):

            async_results.append(
                pool.apply_async(score_batch,
                    args = (
                        i, 
                        batch
                    )
                )
            )

        # Clean up
        pool.close()
        pool.join()

        # Get the results
        results = [async_result.get() for async_result in async_results]
        
        for result in results:
            print(result)


def score_batch(i, batch):
    '''Worker function to score the batch'''

    result = []
    
    for line in batch:
        line = line.split(' ')[:5]
        result.append(f'{i}: {line}')
    
    return result
        