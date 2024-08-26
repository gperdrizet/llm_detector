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

            logger.info(f'Submitting batch {i}')

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


def score_batch(i: int = None, batch: list = None) -> dict:
    '''Worker function to score the batch'''

    ##### Prep the worker & batch ########################################

    # Build empty results dictionary for this batch
    results = {}

    # Loop on variable names from config file
    for var_name in config.DEPENDENT_VARS:

        # Add and empty list for each variable to the results dictionary
        results[var_name] = []

    # Sample text fragments from each record in the batch
    fragments = {
        'Dataset': [],
        'Source': [],
        'String': []
    }

    for record in batch:
        
        # Load the line into a dictionary
        record = json.loads(record)

        # Get the human and synthetic texts
        texts = {}
        texts['human'] = record['Human text']
        texts['synthetic'] = record['Synthetic text']

        # Treat human and synthetic identically

        # Split text to list
        human_text_list = texts['human'].split(' ')
        synthetic_text_list = texts['synthetic'].split(' ')

        # Decide if we should sample ab ovo usque ad mala or vice versa
        reverse_sample = False

        coinflip = random.randint(0,1)

        if coinflip == 1:
            reverse_sample = True
            human_text_list = list(reversed(human_text_list))
            synthetic_text_list = list(reversed(synthetic_text_list))

        # Get the total lengths, choose the shortest of the two
        total_length = min(len(human_text_list), len(synthetic_text_list))

        # Counters for sample edges
        i,j = 0,0

        # Loop until the right edge is past the end
        while j < total_length:

            # Pick a random fragment length

            # If the fragment length is shorter than the long limit, use the
            # fragment length as the upper bound
            long_limit = config.LONG_FRAGMENT_LIMIT

            if long_limit > total_length:
                long_limit = total_length

            slice_length = random.randint(config.SHORT_FRAGMENT_LIMIT, long_limit)

            # Set the slice window
            j = i + slice_length

            # Un-reverse, if needed
            if reverse_sample == True:
                human_text_slice = reversed(human_text_list)
                synthetic_text_slice = reversed(synthetic_text_list)

            # Grab the slices
            human_text_slice = ' '.join(human_text_list[i:j])
            synthetic_text_slice = ' '.join(synthetic_text_list[i:j])

            # Add sampled fragments
            fragments['Dataset'].append(record['Data source'])
            fragments['Source'].append('human')
            fragments['String'].append(human_text_slice)
            
            fragments['Dataset'].append(record['Data source'])
            fragments['Source'].append('synthetic')
            fragments['String'].append(synthetic_text_slice)

    return fragments


    ##### Load the LLMs #################################################
    
    # Set available CPU cores, maxing out based on worker count
    torch.set_num_threads(int(mp.cpu_count()/config.WORKERS))


    result = []
    
    for line in batch:
        line = line.split(' ')[:5]
        result.append(f'{i}: {line}')
    
    return result
        