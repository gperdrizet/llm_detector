'''Collection of function to run each benchmark task'''

from __future__ import annotations
from typing import Callable

import time
import json
import random
import tracemalloc
from random import sample
from multiprocessing import Process, Queue
import torch
import configuration as config
import classes.llm as llm_class
import classes.experiment as experiment_class
from functions.metrics import perplexity, entropy

def benchmark(
    benchmark_func: Callable=None,
    resume: str='False',
    experiment_config_file: str=None,
    logger: Callable=None
) -> None:

    '''Generalized function for benchmarking time or resource utilization'''

    # Instantiate experiment class instance
    experiment=experiment_class.Experiment(
        experiment_config_file,
        logger
    )

    logger.info('Experiment class instance created')

    # Handle resuming from prior data, if needed
    if resume == 'True':
        experiment.resume()
        logger.info('Prior run resumed')

    # Get the independent variable names
    independent_var_names=list(experiment.independent_vars.keys())

    # Create a queue to pass the experiment class instance back and
    # forth between the main loop worker and the benchmark process
    queue=Queue()

    # Loop on conditions
    logger.info('Starting main loop')

    for i, condition in enumerate(experiment.conditions, start=1):

        logger.info(f'Running condition {i} of {len(experiment.conditions)}')

        # Put the experiment class instance in the queue
        queue.put(experiment)

        # Run guts of loop in subprocess
        p=Process(target=main_loop,
            kwargs=dict(
                queue=queue,
                benchmark_func=benchmark_func,
                condition=condition,
                independent_var_names=independent_var_names,
                logger=logger
            )
        )

        p.start()
        p.join()

        # Get the experiment class instance back out of the queue and save the results
        experiment=queue.get()
        experiment.save()


def main_loop(
    queue: Callable=None,
    benchmark_func: Callable=None,
    condition: tuple=None,
    independent_var_names: list=None,
    logger: Callable=None
) -> None:

    '''Guts of the main loop on conditions called by the benchmark function.
    Encapsulated in a function so it can be called in a subprocess to make
    sure that everything incidental to the run, dies with the run.'''

    # Get the experiment class instance from the queue
    experiment=queue.get()

    # Instantiate a new llm class instance
    llm=llm_class.Llm(logger=logger)

    # Loop on the names and values of the independent variables for this run
    for var_name, value in zip(independent_var_names, condition):

        logger.info(f' {var_name}: {value}')

        # Record the values in the experiment class instance
        experiment.independent_vars[var_name].append(value)

        # Set the values in the llm
        if var_name in dir(llm):
            setattr(llm, var_name, value)

    # Call the main benchmark function, catching CUDA errors
    try:
        benchmark_func(experiment, llm)

    # If anything weird happens, print the error and enter appropriate
    # error string in the dependent variables
    except RuntimeError as runtime_error:

        logger.error(f'{runtime_error}')

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string='OOM'

        # For anything else, use NAN
        else:
            error_string='NAN'

        # Enter the error string in all of the dependent variables
        # for this run
        for var_name in experiment.dependent_vars.keys():
            experiment.dependent_vars[var_name].append(error_string)

    # Clean up for next run & put the experiment class instance back
    # into the queue for the benchmark process to retrieve
    llm.clear()
    queue.put(experiment)


def load_time(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main benchmark function to time loading llm and tokenizer'''

    # Time the loading of the model
    loading_start_time = time.time()
    llm.load()
    total_load_time = time.time() - loading_start_time

    # Record the results
    experiment.dependent_vars['load_time'].append(total_load_time)


def generation_rate(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main function to run generation rate benchmark'''

    # Load the model
    llm.load()

    # Time the prompting of the model
    inference_start=time.time()
    _, output_ids=llm.prompt(config.PROMPT)
    total_inference_time=time.time() - inference_start

    # Count tokens generated
    tokens_generated=len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate=tokens_generated / total_inference_time

    # Record the results
    experiment.dependent_vars['tokens_generated'].append(tokens_generated)
    experiment.dependent_vars['inference_time'].append(total_inference_time)
    experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


def decoding_strategy(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main function to run decoding strategy benchmark'''

    # Load the model
    llm.load()

    # Time the prompting of the model
    inference_start=time.time()
    _, output_ids=llm.prompt(config.PROMPT)
    total_inference_time=time.time() - inference_start

    # Count tokens generated
    tokens_generated=len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate=tokens_generated / total_inference_time

    # Record the results
    experiment.dependent_vars['tokens_generated'].append(tokens_generated)
    experiment.dependent_vars['inference_time'].append(total_inference_time)
    experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


def encoding_memory(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main function to run encoding memory benchmark'''

    # Sample the test text
    text_list=config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample=sample(
        text_list,
        experiment.independent_vars['input_length'][-1]
    )

    input_text=' '.join(text_list_sample)

    # Load the model
    llm.load()

    # Reset memory stats for all devices
    for device in config.AVAILABLE_GPUS:
        torch.cuda.reset_peak_memory_stats(device=device)

    # Time the encoding
    encoding_start=time.time()

    # Encode
    encodings=llm.tokenizer(
        input_text,
        return_tensors="pt",
        return_token_type_ids=False
    ).to('cuda')

    encoding_time=time.time() - encoding_start

    # Get encoded fragment length
    fragment_length=encodings['input_ids'].shape[1]

    # Get encoding rate
    encoding_rate=fragment_length / encoding_time

    # Get total peak memory
    peak_memory=0

    for device in config.AVAILABLE_GPUS:
        peak_memory += torch.cuda.max_memory_allocated(device=device) / (10 ** 9)

    # Record the results
    experiment.dependent_vars['peak_memory'].append(peak_memory)
    experiment.dependent_vars['tokens'].append(fragment_length)
    experiment.dependent_vars['encoding_time'].append(encoding_time)
    experiment.dependent_vars['encoding_rate'].append(encoding_rate)

def logits_memory(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main function to run logits memory benchmark'''

    # Sample the test text
    text_list=config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample=sample(
        text_list,
        experiment.independent_vars['input_length'][-1]
    )

    input_text=' '.join(text_list_sample)

    # Load the model
    llm.load()

    # Encode
    encodings=llm.tokenizer(
        input_text,
        return_tensors="pt",
        return_token_type_ids=False
    ).to('cuda')

    # Get encoded fragment length
    fragment_length=encodings['input_ids'].shape[1]

    # Reset memory stats for all devices
    for device in config.AVAILABLE_GPUS:
        torch.cuda.reset_peak_memory_stats(device=device)

    # Time the logits calculation
    logits_start=time.time()
    _=llm.model(**encodings).logits
    logits_time=time.time() - logits_start

    # Get calculation rate
    rate=fragment_length / logits_time

    # Get total peak memory
    peak_memory=0

    for device in config.AVAILABLE_GPUS:
        peak_memory += torch.cuda.max_memory_allocated(device=device) / (10 ** 9)

    # Record the results
    experiment.dependent_vars['peak_memory'].append(peak_memory)
    experiment.dependent_vars['tokens'].append(fragment_length)
    experiment.dependent_vars['logits_time'].append(logits_time)
    experiment.dependent_vars['rate'].append(rate)


def logits_cpu(
    experiment: Callable=None,
    llm: Callable=None
) -> None:

    '''Main function to run logits cpu benchmark'''

    # Sample the test text
    text_list=config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample=sample(
        text_list,
        experiment.independent_vars['input_length'][-1]
    )

    input_text=' '.join(text_list_sample)

    # Start tracking system memory
    if experiment.independent_vars['device_map'][-1] == 'cpu':
        tracemalloc.start()

    # Load the model
    llm.load()

    # Encode
    encodings=llm.tokenizer(
        input_text,
        return_tensors="pt",
        return_token_type_ids=False
    )

    if experiment.independent_vars['device_map'][-1] != 'cpu':
        encodings=encodings.to('cuda')

    # Get encoded fragment length
    fragment_length=encodings['input_ids'].shape[1]

    if experiment.independent_vars['device_map'][-1] != 'cpu':

        # Reset memory stats for all GPUs
        for device in config.AVAILABLE_GPUS:
            torch.cuda.reset_peak_memory_stats(device=device)

    # Time the logits calculation
    logits_start=time.time()
    _=llm.model(**encodings).logits
    logits_time=time.time() - logits_start

    # Get calculation rate
    rate=fragment_length / logits_time

    # Get total peak GPU memory
    if experiment.independent_vars['device_map'][-1] != 'cpu':
        max_memory=0

        for device in config.AVAILABLE_GPUS:
            max_memory += torch.cuda.max_memory_allocated(device=device) / (10 ** 9)

    # Get peak system memory
    if experiment.independent_vars['device_map'][-1] == 'cpu':

        _, max_memory = tracemalloc.get_traced_memory()
        max_memory = max_memory / (10 ** 6)
        tracemalloc.stop()

    # Record the results
    experiment.dependent_vars['max_memory'].append(max_memory)
    experiment.dependent_vars['tokens'].append(fragment_length)
    experiment.dependent_vars['logits_time'].append(logits_time)
    experiment.dependent_vars['rate'].append(rate)


def binoculars_model(
    experiment: Callable=None,
    observer_model: Callable=None
) -> None:

    '''Main function to run binoculars score benchmark'''

    # Load the data
    input_file=f'{config.BINOCULARS_DATA_PATH}/aggregated_hans_data.json'

    with open(input_file, encoding="utf-8") as file:
        data=json.load(file)

    # Find out how many records we have for use later
    num_records=len(list(data.keys()))

    # Set reuseable devices - using both chips on a single K80 for now
    observer_device='cuda:1'
    performer_device='cuda:2'

    # Set available CPU cores - doing this from the LLM class does not seem to work
    torch.set_num_threads(16)

    # Load two instances of the model, one we already have from the main loop so just
    # make sure it goes to the right device before loading it. The other we need to instantiate.
    observer_model.device_map=observer_device

    # Pick the correct instruct model based on the observer model we are running
    if observer_model.hf_model_string == 'meta-llama/Meta-Llama-3-8B':
        performer_model_hf_string='meta-llama/Meta-Llama-3-8B-instruct'

    elif observer_model.hf_model_string == 'tiiuae/falcon-7b':
        performer_model_hf_string='tiiuae/falcon-7b-instruct'

    elif observer_model.hf_model_string == 'mistralai/Mistral-7B-v0.3':
        performer_model_hf_string='mistralai/Mistral-7B-Instruct-v0.3'

    elif observer_model.hf_model_string == "meta-llama/Llama-2-7b-hf":
        performer_model_hf_string='meta-llama/Llama-2-7b-chat-hf'

    performer_model=llm_class.Llm(
        hf_model_string=performer_model_hf_string,
        device_map=performer_device
    )

    # Load the models
    observer_model.load()
    performer_model.load()

    # Set the models to evaluation mode to deactivate any dropout modules
    # the is done to ensure reproducibility of results during evaluation
    observer_model.model.eval()
    performer_model.model.eval()

    # Add end of sequence for the pad token if one has not been defined
    if not observer_model.tokenizer.pad_token:
        observer_model.tokenizer.pad_token=observer_model.tokenizer.eos_token

    # Sample the data 1000 times
    fragment_count=0
    texts={}

    while fragment_count < 200:

        # Pick a random record number
        record_id=random.randint(0, num_records - 1)

        # Pull the record and get the human and synthetic texts
        record=data[str(record_id)]
        texts['human']=record['Human text']
        texts['synthetic']=record['Synthetic text']

        # Score both
        for text_source, text in texts.items():

            # Split text to list
            text_list=text.split(' ')

            # Get the total length
            total_length=len(text_list)

            # Set counters for the fragment start and end
            i,j=0,0

            # Loop until the right edge is past the end
            while j < total_length:

                # Count the fragment
                fragment_count+=1

                # Pick a random length between 50 and 500 tokens
                slice_length=random.randint(100, 300)

                # If the slice length is greater than the length
                # of the input tokens, use all of them
                if slice_length > total_length:
                    slice_length=total_length

                # Set the window
                j=i + slice_length

                # Grab the slice
                text_list_slice=text_list[i:j]

                # Make it a string
                text_string_slice=' '.join(text_list_slice)

                # Fence to catch CUDA OOM
                try:
                    # Encode
                    encodings=observer_model.tokenizer(
                        text_string_slice,
                        return_tensors="pt",
                        return_token_type_ids=False
                    ).to(observer_device)

                    # Get input ids only as list for later logging/data collection
                    fragment_length_tokens=encodings['input_ids'].shape[1]

                    # Calculate logits
                    observer_logits=observer_model.model(**encodings).logits
                    performer_logits=performer_model.model(**encodings).logits

                    observer_model.logger.info('Slice encoded')
                    observer_model.logger.info('Encoded slice length: %s', fragment_length_tokens)
                    observer_model.logger.info('Logits length: %s', {performer_logits.shape})

                    ppl=perplexity(encodings, performer_logits)
                    observer_model.logger.info('Have slice perplexity')

                    x_ppl=entropy(
                        observer_logits.to('cuda:0'),
                        performer_logits.to('cuda:0'),
                        encodings.to('cuda:0'),
                        observer_model.tokenizer.pad_token_id
                    )

                    observer_model.logger.info('Have cross perplexity')

                    binoculars_scores = ppl / x_ppl
                    binoculars_scores = binoculars_scores.tolist()
                    observer_model.logger.info('Binoculars score: %s', binoculars_scores[0])

                except RuntimeError as runtime_error:

                    observer_model.logger.error(runtime_error)

                    # For out of memory enter OOM
                    if 'CUDA out of memory' in str(runtime_error):
                        error_string='OOM'

                    # Otherwise enter NAN:
                    else:
                        error_string='NAN'

                    ppl=[error_string]
                    x_ppl=[error_string]
                    binoculars_scores=[error_string]

                # Record the results
                experiment.dependent_vars['binoculars_score'].append(str(binoculars_scores[0]))
                experiment.dependent_vars['perplexity'].append(str(ppl[0]))
                experiment.dependent_vars['cross-perplexity'].append(str(x_ppl[0]))
                experiment.dependent_vars['length_words'].append(slice_length)
                experiment.dependent_vars['length_tokens'].append(fragment_length_tokens)
                experiment.dependent_vars['data_source'].append(record['Data source'])
                experiment.dependent_vars['generating_model'].append(record['Generation model'])

                if text_source == 'human':
                    experiment.dependent_vars['human_text'].append(True)
                else:
                    experiment.dependent_vars['human_text'].append(False)

                # Reset for the next loop
                i=j

