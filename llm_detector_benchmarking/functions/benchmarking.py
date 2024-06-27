'''Collection of function to run each benchmark task'''

from __future__ import annotations
from typing import Callable

import time
import tracemalloc
from random import sample
from multiprocessing import Process, Queue
import torch
import configuration as config
import classes.llm as llm_class
import classes.experiment as experiment_class

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
