'''Second generation functions for perplexity ratio scoring 
of data from https://arxiv.org/abs/2401.12070'''

from __future__ import annotations
from typing import Callable

import os
import time
import json
import random
import torch
import logging
import tracemalloc
import multiprocessing as mp
from pathlib import Path

import benchmarking.configuration as config
import benchmarking.functions.helper as helper_funcs
import benchmarking.classes.llm as llm_class
from benchmarking.functions.metrics import perplexity, entropy

os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    results_datafile = f'{config.HANS_DATA_PATH}/{output_file_name}.json'

    results, record_number = initialize_results(results_datafile)

    # Open the data so we can loop on the lines
    with open(input_datafile, encoding = 'utf-8') as f:

        # Skip already completed records
        for i in range(record_number):
            next_line = f.readline()
            logger.info(f'Skipped record {i}, already sampled')

        # Set flag to detect EOF
        next_line = 'next line'

        # Loop until we break
        while True:

            # Make a set of batches of lines from the input data
            record_number, batches = make_batches(f, record_number)

            # If batches came back empty, we are done.
            if len(batches) == 0:
                break

            # If we have batches, run the scoring functions
            if len(batches) != 0:

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
                
                ##### Collect and save the results #########################################

                # Get the results
                new_results = [async_result.get() for async_result in async_results]

                # Add the new results
                for new_result in new_results:
                    for key, value in new_result.items():
                        results[key].extend(value)

                # Save
                with open(results_datafile, 'w', encoding = 'utf-8') as out_f:
                    json.dump(results, out_f)


def score_batch(worker_num: int = None, batch: list = None) -> dict:
    '''Worker function to calculate perplexity ratio score for each
    fragment in the batch'''

    # Get the logger
    logger = logging.getLogger(__name__)

    # Build empty results dictionary for this batch
    results = make_empty_results_dict()

    ##### Sample text fragments from each record in the batch ##########
    logger.info(f'Worker {worker_num} - Sampling from batch')

    # Loop on each record in the batch
    for element in batch:
        
        # Get the record number and content
        record_number = element['record_number']
        record = element['record']

        # Load the record content into a dictionary
        record = json.loads(record)

        # Get the human and synthetic texts
        texts = {}
        texts['human'] = record['Human text']
        texts['synthetic'] = record['Synthetic text']

        ##### Generate text fragment samples from the batch's human 
        ##### and synthetic text and add them to the results
        results = make_text_fragment_samples(
            texts, 
            worker_num, 
            record_number, 
            record, 
            results
        )
    
    num_samples = len(results['Dataset'])
    logger.info(f'Worker {worker_num} - Generated {num_samples} samples from record')

    ##### Calculate perplexity scores for each fragment from batch

    # Set available CPU cores based on worker count
    cpus = int(mp.cpu_count()/config.WORKERS) - 2
    torch.set_num_threads(cpus)
    logger.info(f'Worker {worker_num} - {cpus} CPU threads avalible')

    # Load the models
    reader_model, writer_model = load_models()
    logger.info(f'Worker {worker_num} - Reader and writer models ready')

    ##### Fragment loop #######################################################

    # Loop on sampled fragments using a counter to scan
    # and add to the results
    for i in range(len(results['Dataset'])):

        # Fence to catch CUDA OOM
        try:

            ##### Do the reader things ############################################

            # Start memory tracking
            start_memory_tracking(config.READER_DEVICE)

            # Start timer
            start_time = time.time()

            # Encode
            encodings = reader_model.tokenizer(
                results['String'][i],
                return_tensors = 'pt',
                return_token_type_ids = False
            ).to(config.READER_DEVICE)

            # Get reader logits
            reader_logits = reader_model.model(**encodings).logits

            # Stop timer
            reader_dT = time.time() - start_time

            logger.info(f'Worker {worker_num} - Reader encoded fragment {i + 1} of {num_samples}')

            # Get length of input in tokens
            fragment_length_tokens = encodings['input_ids'].shape[1]

            # Get reader peak memory in GB
            reader_peak_memory = collect_memory_data(config.READER_DEVICE)

            ##### Do the writer things ######################################################

            # Start memory tracking
            start_memory_tracking(config.WRITER_DEVICE)

            # Start timer
            start_time = time.time()

            # Get the writer logits
            writer_logits = writer_model.model(**encodings).logits

            # Stop timer
            writer_dT = time.time() - start_time

            logger.info(f'Worker {worker_num} - Writer encoded fragment {i + 1} of {num_samples}')

            # Get writer peak memory in GB
            writer_peak_memory = collect_memory_data(config.WRITER_DEVICE)

            ##### Do the perplexity things #########################################################

            ppl = perplexity(encodings, writer_logits)

            x_ppl = entropy(
                reader_logits,
                writer_logits.to(config.READER_DEVICE),
                encodings,
                reader_model.tokenizer.pad_token_id
            )

            perplexity_ratio_score = ppl / x_ppl

        except RuntimeError as runtime_error:

            logger.error(runtime_error)

            # For out of memory enter OOM
            if 'CUDA out of memory' in str(runtime_error):
                error_string = 'OOM'

            # Otherwise enter NAN:
            else:
                error_string = 'NAN'

            # Set variables we didn't get values for to the error string
            reader_dT = error_string
            writer_dT = error_string
            fragment_length_tokens = error_string
            reader_peak_memory = error_string
            writer_peak_memory = error_string
            ppl = [error_string]
            x_ppl = [error_string]
            perplexity_ratio_score = [error_string]

        # Add everything to results
        results['Reader time (seconds)'].append(str(reader_dT))
        results['Writer time (seconds)'].append(str(writer_dT))
        results['Fragment length (tokens)'].append(str(fragment_length_tokens))
        results['Reader peak memory (GB)'].append(str(reader_peak_memory))
        results['Writer peak memory (GB)'].append(str(writer_peak_memory))
        results['Perplexity'].append(str(ppl[0]))
        results['Cross-perplexity'].append(str(x_ppl[0]))
        results['Perplexity ratio score'].append(str(perplexity_ratio_score[0]))
    
    return results


def initialize_results(results_datafile: str = None):
    '''Sets up results data struct for run and handles resuming from
    prior data if any exists.'''

    # If the results file exists, load the data into results
    if Path(results_datafile).is_file():
        with open(results_datafile, encoding = 'utf-8') as f:
            results = json.load(f)

    # If we don't already have results, initialize an empty results
    # dictionary
    else:

        results = {}

        # Loop on variable names from config file
        for var_name in config.DEPENDENT_VARS:

            # Add and empty list for each variable to the results dictionary
            results[var_name] = []

    # Get the list of record numbers we have already sampled in the data file
    # so that we don't repeat them
    sampled_record_numbers = list(set(results['Source record num']))
    sampled_record_numbers = list(map(int, sampled_record_numbers))

    # Initialize the record number to the highest already completed 
    # sample, or zero if we don't have prior data
    if len(sampled_record_numbers) > 0:
        record_number = max(sampled_record_numbers) + 1

    else:
        record_number = 0

    return results, record_number

def make_batches(f, record_number):
    '''Makes set of batches from input data records'''

    # Get the logger
    logger = logging.getLogger(__name__)

    # Make a set of batches of lines from the input data
    batches = []

    # Loop on worker count to make a batch for each one
    for _ in range(config.WORKERS):

        # Make a batch
        batch = []

        # Loop on batch size
        for _ in range(config.BATCH_SIZE):

            # Read next line from input file and add to batch
            next_line = f.readline()

            if next_line == '':
                break

            batch.append({'record_number': record_number, 'record': next_line})
            record_number += 1
        
        if next_line == '':
            logger.info('End of input data file')
            break

        # Once the batch is complete, add it to the batches
        batches.append(batch)

    return record_number, batches


def make_empty_results_dict():
    '''Uses variable names in configuration file to build empty
    dictionary to hold results from batch'''

    # Empty dictionary to hold results
    results = {}

    # Loop on variable names from config file
    for var_name in config.DEPENDENT_VARS:

        # Add and empty list for each variable to the results dictionary
        results[var_name] = []

    return results


def make_text_fragment_samples(
        texts: dict = None, 
        worker_num: int = None, 
        record_number: int = None, 
        record: dict = None, 
        results: dict = None
) -> dict:

    '''Generates a text fragment samples of random length from
    human and synthetic text in input data record'''

    # Get the logger
    logger = logging.getLogger(__name__)

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

    human_text_length = len(human_text_list)
    synthetic_text_length = len(synthetic_text_list)

    logger.info(f'Worker {worker_num} - Total human text: {human_text_length} words')
    logger.info(f'Worker {worker_num} - Total synthetic text: {synthetic_text_length} words')

    # Get the total lengths, choose the shortest of the two
    total_length = min(human_text_length, synthetic_text_length)
    logger.info(f'Worker {worker_num} - Apparent length: {total_length} words')

    # Make sure the fragment is at least as long as the short limit
    if total_length > config.SHORT_FRAGMENT_LIMIT:

        # Counters for sample edges
        i,j = 0,0

        # Loop until the right edge is past the end
        sample_count = 1

        while j < total_length:

            # Pick a random fragment length

            # If the fragment length is shorter than the long limit, use the
            # fragment length as the upper bound
            long_limit = config.LONG_FRAGMENT_LIMIT

            if long_limit > total_length:
                long_limit = total_length

            slice_length = random.randint(config.SHORT_FRAGMENT_LIMIT, long_limit)
            logger.info(f'Worker {worker_num} - Sample {sample_count} - Sample fragment length: {slice_length} words')

            # Set the slice window
            j = i + slice_length

            # Loop until the right edge of the slice window ends up past the end of the text
            if j <= total_length:

                # Get the slices
                human_text_slice = human_text_list[i:j]
                synthetic_text_slice = synthetic_text_list[i:j]

                # Get true lengths in words
                human_text_length_words = len(human_text_slice)
                synthetic_text_length_words = len(synthetic_text_slice)

                logger.info(f'Worker {worker_num} - Sample {sample_count} - Human fragment length: {human_text_length_words} words')
                logger.info(f'Worker {worker_num} - Sample {sample_count} - Synthetic fragment length: {synthetic_text_length_words} words')

                # Reset for the next loop
                i = j
                sample_count += 1

                # If we reversed the string to sample from the end, reverse it
                # back so the result is always the same strand
                if reverse_sample == True:
                    human_text_slice = list(reversed(human_text_slice))
                    synthetic_text_slice = list(reversed(synthetic_text_slice))

                # Make strings
                human_text_string = ' '.join(human_text_slice)
                synthetic_text_string = ' '.join(synthetic_text_slice)

                # Add everything to the results
                results['Source record num'].append(str(record_number))
                results['Dataset'].append(record['Data source'])
                results['Source'].append('human')
                results['Fragment length (words)'].append(str(human_text_length_words))
                results['String'].append(human_text_string)

                results['Source record num'].append(str(record_number))
                results['Dataset'].append(record['Data source'])
                results['Source'].append('synthetic')
                results['Fragment length (words)'].append(str(synthetic_text_length_words))
                results['String'].append(synthetic_text_string)
            
            else:
                logger.info(f'Worker {worker_num} - Sample {sample_count} - Remaining text too short for sample')

    return results


def load_models():
    '''Loads and initializes reader and writer model'''

    # Load the models
    reader_model = llm_class.Llm(
        hf_model_string = config.READER_MODEL,
        device_map = config.READER_DEVICE
    )
    writer_model = llm_class.Llm(
        hf_model_string = config.WRITER_MODEL,
        device_map = config.WRITER_DEVICE
    )

    reader_model.load()
    writer_model.load()

    # Set the models to evaluation mode to deactivate any dropout
    # modules to ensure reproducibility of results during evaluation
    reader_model.model.eval()
    writer_model.model.eval()

    # Add end of sequence for the pad token if one has not been defined
    if not reader_model.tokenizer.pad_token:
        reader_model.tokenizer.pad_token = reader_model.tokenizer.eos_token

    return reader_model, writer_model


def start_memory_tracking(device: str = None):
    '''Initializes memory tracking using the appropriate method based on
    the device specified in the configuration file'''

    if 'cuda' in device:
        torch.cuda.reset_peak_memory_stats(device = device)

    elif device == 'cpu':
        tracemalloc.start()

    return


def collect_memory_data(device: str = None) -> float:
    '''Collects and returns peak memory use using an
    appropriate strategy based on the device'''

    # Get reader peak memory in GB
    if 'cuda' in device:
        peak_memory = torch.cuda.max_memory_allocated(device = device)
        peak_memory = peak_memory / (10 ** 9)

    elif device == 'cpu':
        _, max_memory = tracemalloc.get_traced_memory()
        peak_memory = max_memory * (1.024 * (10**-3))
        tracemalloc.stop()

    return peak_memory