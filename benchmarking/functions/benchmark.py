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
import benchmarking.configuration as config
import benchmarking.classes.llm as llm_class
import benchmarking.classes.experiment as experiment_class
from benchmarking.functions.metrics import perplexity, entropy

# Comment ##############################################################
# Code ########################################################################

def run(
        resume: str = 'False',
        experiment_config_file: str = None
) -> None:

    '''Generalized function for benchmarking experiments'''

    # Instantiate experiment class instance
    experiment = experiment_class.Experiment(
        experiment_config_file
    )

    experiment.logger.info('Starting %s', experiment.experiment_name)
    experiment.logger.info('Experiment class instance created')

    # Handle resuming from prior data, if needed
    if resume == 'True':
        experiment.resume()
        experiment.logger.info('Prior run resumed')

    # Get the independent variable names
    independent_var_names = list(experiment.independent_vars.keys())

    # Find the index of the iteration number so we can pull it from the
    # condition tuple
    iteration_index = independent_var_names.index('iteration')

    # Create a queue to pass the experiment class instance back and
    # forth between the main loop worker and the benchmark process
    queue = Queue()

    # Holder to collect iterations for batch
    conditions_batch = []

    experiment.logger.info('Starting main loop')

    # Loop on conditions
    for i, condition in enumerate(experiment.conditions):

        # Tracker variables
        last_condition = False
        batch_complete = False

        # If we are at the last condition, mark it and set trigger
        # the last run by setting batch_complete to true
        if i + 1 == len(experiment.conditions):
            last_condition = True
            batch_complete = True
            experiment.logger.info('Last condition of experiment')

        # If this is not the last condition, check to see if the
        # next condition's iteration number is 1, if it is, the
        # current condition is the last one of this batch
        elif i + 1 < len(experiment.conditions):
            if experiment.conditions[i + 1][iteration_index] == 1:
                batch_complete = True
                experiment.logger.info('Last condition of batch')

        # If the batch is not complete, add the current condition
        # to the batch and move on
        if batch_complete is False:

            conditions_batch.append(condition)
            experiment.logger.info('Added condition %s of %s to bach',
                                i + 1, len(experiment.conditions))

        # If this is the last condition, or the batch is complete
        # add the current condition to the batch and run the batch
        elif last_condition is True or batch_complete is True:

            conditions_batch.append(condition)
            experiment.logger.info('Added condition %s of %s to bach',
                                i + 1, len(experiment.conditions))
            experiment.logger.info('Running batch of %s conditions:',
                                   len(conditions_batch))

            # Put the experiment class instance in the queue
            queue.put(experiment)

            # Run guts of loop in subprocess
            p = Process(target = run_batch,
                kwargs = dict(
                    queue = queue,
                    conditions_batch = conditions_batch,
                    independent_var_names = independent_var_names
                )
            )

            # Run the job
            p.start()
            p.join()

            # Get the experiment class instance back out of the queue
            # and save the results
            experiment = queue.get()
            experiment.save()

            # Reset the batch
            conditions_batch = []

    experiment.logger.info('%s run complete', experiment.experiment_name)


def run_batch(
        queue: Callable=None,
        conditions_batch: tuple=None,
        independent_var_names: list=None,
) -> None:

    '''Guts of the main loop on conditions called by the benchmark 
    function. Encapsulated in a function so it can be called in a 
    subprocess to make sure that everything incidental to the run, dies
    with the run.'''

    # Get the experiment class instance from the queue
    experiment=queue.get()

    # Instantiate a new llm class instance
    llm=llm_class.Llm(logger=experiment.logger)

    # Set the llm parameters - since all of the conditions in this
    # batch are the same except for the iteration number, we can
    # use the first one as the source for our llm parameters
    for var_name, value in zip(independent_var_names, conditions_batch[0]):
        if var_name in dir(llm):
            setattr(llm, var_name, value)

    # The binoculars benchmark needs special handling here because
    # it uses two models - if that's what we are running, set-up a
    # second model
    if experiment.experiment_name == 'binoculars_model_benchmark':
        performer_hf_model_string = choose_binoculars_performer_model(llm)
        performer = llm_class.Llm()
        performer.hf_model_string = performer_hf_model_string
        performer.device_map = 'cuda:2'

        # Also, load the Hans et al. (2024) data while we are at it
        input_file = f'{config.BINOCULARS_DATA_PATH}/aggregated_hans_data.json'

        with open(input_file, encoding = 'utf-8') as file:
            data = json.load(file)

    # Load the llm, catching CUDA errors
    try:
        llm.load()

        # And load the performer for binoculars benchmarks
        if experiment.experiment_name == 'binoculars_model_benchmark':
            performer.load()

    # If anything weird happens, we need to skip this batch. Log the
    # error and enter appropriate error string in the dependent
    # variables then return
    except RuntimeError as runtime_error:

        experiment.logger.error(' Batch failed:')
        experiment.logger.error(f' {runtime_error}')

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string='OOM'

        # For anything else, use NAN
        else:
            error_string='NAN'

        # Loop on the conditions in this batch
        for i, condition in enumerate(conditions_batch, start=1):

            # Loop on the names and values of the independent variables
            # for this run
            for var_name, value in zip(independent_var_names, condition):

                # Record the values in the experiment class instance
                experiment.independent_vars[var_name].append(value)

            # Enter the error string in all of the dependent variables
            # for this run
            for var_name in experiment.dependent_vars.keys():
                experiment.dependent_vars[var_name].append(error_string)

        # Call off the run
        queue.put(experiment)
        return

    # Loop on the conditions in this batch
    for i, condition in enumerate(conditions_batch, start=1):

        experiment.logger.info(f' Batch condition {i}')

        # Loop on the independent variables names and values for this run
        for var_name, value in zip(independent_var_names, condition):

            experiment.logger.info(f'  {var_name}: {value}')

            # Record the values in the experiment class instance
            experiment.independent_vars[var_name].append(value)

        # Call the run specific benchmark function, catching CUDA errors
        try:

            # The binoculars benchmark needs special handling here because
            # it uses two models, and takes data so we need a different
            # function call
            if experiment.experiment_name == 'binoculars_model_benchmark':

                experiment.benchmark_func(
                    experiment = experiment,
                    data = data,
                    observer_model = llm,
                    performer_model = performer
                )

            # For all other benchmarks, use the standard function call
            else:
                experiment.benchmark_func(experiment, llm)

        # If anything weird happens, print the error and enter appropriate
        # error string in the dependent variables
        except RuntimeError as runtime_error:

            experiment.logger.error('  Batch condition failed:')
            experiment.logger.error(f'  {runtime_error}')

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

    # Clean up for next batch & put the experiment class instance back
    # into the queue for the benchmark process to retrieve
    llm.clear()
    queue.put(experiment)


def model_loading_benchmark(
        experiment: Callable = None,
        llm: Callable = None
) -> None:

    '''Main benchmark function to time loading llm and tokenizer'''

    # Since we are benchmarking the load time here - we need to clear
    # then llm so we can reload it while timing. Not great, since doing
    # it this way causes an extra load per batch, but this set-up is
    # much better for the many other benchmarks that use this test
    # harness
    llm.clear()

    # Time the loading of the model
    loading_start_time = time.time()
    llm.load()
    total_load_time = time.time() - loading_start_time

    # Record the results
    experiment.dependent_vars['load_time'].append(total_load_time)


def generation_rate_benchmark(
        experiment: Callable = None,
        llm: Callable = None
) -> None:

    '''Main function to run generation rate benchmark'''

    # Time the prompting of the model
    inference_start = time.time()
    _, output_ids = llm.prompt(config.PROMPT)
    total_inference_time = time.time() - inference_start

    # Count tokens generated
    tokens_generated = len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate = tokens_generated / total_inference_time

    # Record the results
    experiment.dependent_vars['tokens_generated'].append(tokens_generated)
    experiment.dependent_vars['inference_time'].append(total_inference_time)
    experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


def decoding_strategy_benchmark(
        experiment: Callable = None,
        llm: Callable = None
) -> None:

    '''Main function to run decoding strategy benchmark'''

    # Time the prompting of the model
    inference_start = time.time()
    _, output_ids = llm.prompt(config.PROMPT)
    total_inference_time = time.time() - inference_start

    # Count tokens generated
    tokens_generated = len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate = tokens_generated / total_inference_time

    # Record the results
    experiment.dependent_vars['tokens_generated'].append(tokens_generated)
    experiment.dependent_vars['inference_time'].append(total_inference_time)
    experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


def encoding_memory_benchmark(
        experiment: Callable = None,
        llm: Callable = None
) -> None:

    '''Main function to run encoding memory benchmark'''

    # Sample the test text
    text_list = config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample = sample(
        text_list,
        experiment.independent_vars['input_length'][-1]
    )

    input_text = ' '.join(text_list_sample)

    # Reset memory stats for all devices
    for device in config.AVAILABLE_GPUS:
        torch.cuda.reset_peak_memory_stats(device = device)

    # Time the encoding
    encoding_start = time.time()

    # Encode
    encodings = llm.tokenizer(
        input_text,
        return_tensors = 'pt',
        return_token_type_ids = False
    ).to('cuda')

    encoding_time = time.time() - encoding_start

    # Get encoded fragment length
    fragment_length = encodings['input_ids'].shape[1]

    # Get encoding rate
    encoding_rate=fragment_length / encoding_time

    # Get total peak memory
    peak_memory = 0

    for device in config.AVAILABLE_GPUS:
        peak_memory += torch.cuda.max_memory_allocated(device = device) / (10 ** 9)

    # Record the results
    experiment.dependent_vars['peak_memory'].append(peak_memory)
    experiment.dependent_vars['tokens'].append(fragment_length)
    experiment.dependent_vars['encoding_time'].append(encoding_time)
    experiment.dependent_vars['encoding_rate'].append(encoding_rate)


def logits_calculation_benchmark(
        experiment: Callable = None,
        llm: Callable = None
) -> None:

    '''Main function to run logits cpu benchmark'''

    # Sample the test text
    text_list = config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample = sample(
        text_list,
        experiment.independent_vars['input_length'][-1]
    )

    input_text=' '.join(text_list_sample)

    # Encode
    encodings = llm.tokenizer(
        input_text,
        return_tensors = 'pt',
        return_token_type_ids = False
    )

    # If this is not a CPU run, move encoding to GPU
    if experiment.independent_vars['device_map'][-1] != 'cpu':
        encodings = encodings.to('cuda')

    # Get encoded fragment length
    fragment_length = encodings['input_ids'].shape[1]

    # Start memory tracking using the correct strategy based on device map
    if experiment.independent_vars['device_map'][-1] != 'cpu':

        # Reset memory stats for all GPUs
        for device in config.AVAILABLE_GPUS:
            torch.cuda.reset_peak_memory_stats(device = device)

    elif experiment.independent_vars['device_map'][-1] == 'cpu':
        tracemalloc.start()

    # Time the logits calculation
    logits_start = time.time()
    _ = llm.model(**encodings).logits
    logits_time = time.time() - logits_start

    # Get calculation rate
    rate=fragment_length / logits_time

    # Get max memory using the correct strategy based on device map
    if experiment.independent_vars['device_map'][-1] != 'cpu':
        max_memory = 0

        for device in config.AVAILABLE_GPUS:
            device_max_memory = torch.cuda.max_memory_allocated(device=device)
            device_max_memory = device_max_memory / (10 ** 9)
            max_memory += device_max_memory

    elif experiment.independent_vars['device_map'][-1] == 'cpu':

        _, max_memory = tracemalloc.get_traced_memory()
        max_memory = max_memory / (10 ** 6)
        tracemalloc.stop()

    # Record the results
    experiment.dependent_vars['max_memory'].append(max_memory)
    experiment.dependent_vars['tokens'].append(fragment_length)
    experiment.dependent_vars['logits_time'].append(logits_time)
    experiment.dependent_vars['rate'].append(rate)

def choose_binoculars_performer_model(observer_model: Callable=None) -> str:
    '''Picks the correct instruct model to server as the binoculars
    score performer based on the identity of the observer model'''

    if observer_model.hf_model_string == 'meta-llama/Meta-Llama-3-8B':
        performer_model_hf_string='meta-llama/Meta-Llama-3-8B-instruct'

    elif observer_model.hf_model_string == 'tiiuae/falcon-7b':
        performer_model_hf_string='tiiuae/falcon-7b-instruct'

    elif observer_model.hf_model_string == 'mistralai/Mistral-7B-v0.3':
        performer_model_hf_string='mistralai/Mistral-7B-Instruct-v0.3'

    elif observer_model.hf_model_string == 'meta-llama/Llama-2-7b-hf':
        performer_model_hf_string='meta-llama/Llama-2-7b-chat-hf'

    elif observer_model.hf_model_string == 'google/gemma-2-9b':
        performer_model_hf_string='google/gemma-2-9b-it'

    elif observer_model.hf_model_string == 'google/recurrentgemma-2b':
        performer_model_hf_string='google/recurrentgemma-2b-it'

    elif observer_model.hf_model_string == 'Qwen/Qwen2-7B':
        performer_model_hf_string='Qwen/Qwen2-7B-Instruct'

    return performer_model_hf_string


def binoculars_model_benchmark(
        experiment: Callable = None,
        data: dict = None,
        observer_model: Callable = None,
        performer_model: Callable = None
) -> None:

    '''Main function to run binoculars score benchmark'''

    # Set the models to evaluation mode to deactivate any dropout
    # modules the is done to ensure reproducibility of results during
    # evaluation
    observer_model.model.eval()
    performer_model.model.eval()

    # Add end of sequence to the observer's tokenizer for the pad
    # token if not defined
    if not observer_model.tokenizer.pad_token:
        observer_model.tokenizer.pad_token = observer_model.tokenizer.eos_token

    # Find out how many records we have for use later
    num_records = len(list(data.keys()))

    # Sample the data, repeating the sampling until
    # we get a valid text fragment
    text_fragment_string = None

    while text_fragment_string is None:

        # Pick a random record number
        record_id = random.randint(0, num_records - 1)

        # Pull the record and get the human and synthetic texts
        record = data[str(record_id)]
        texts = {'human': record['Human text']}
        texts['synthetic'] = record['Synthetic text']

        # Randomly choose human or synthetic text from this record
        choices = ['human', 'synthetic']
        choice = random.choice(choices)
        text = texts[choice]

        # Split text to list
        text_list = text.split(' ')

        # Get the total length
        total_length = len(text_list)

        # Select random list index for fragment start
        fragment_start = random.randint(0, total_length - 1)

        # Pick a random length between 50 and 300 tokens
        fragment_length = random.randint(50, 300)

        # Grab the slice
        text_fragment_list = text_list[fragment_start:fragment_start + fragment_length]

        # Get the actual fragment length
        fragment_length = len(text_fragment_list)
        observer_model.logger.info('  Fragment length: %s',
                                    fragment_length)
        # Make it a string
        text_fragment_string = ' '.join(text_fragment_list)

    # Fence to catch CUDA OOM
    try:
        # Encode
        encodings = observer_model.tokenizer(
            text_fragment_string,
            return_tensors = 'pt',
            return_token_type_ids = False
        ).to(observer_model.device_map)

        # Get input ids as list for logging/data collection
        fragment_length_tokens = encodings['input_ids'].shape[1]

        # Calculate logits
        observer_logits = observer_model.model(**encodings).logits
        performer_logits = performer_model.model(**encodings).logits

        observer_model.logger.info('  Fragment encoded')
        observer_model.logger.info('  Encoded fragment length: %s',
                                    fragment_length_tokens)
        observer_model.logger.info('  Logits length: %s',
                                    performer_logits.shape[1])

        ppl = perplexity(encodings, performer_logits)
        observer_model.logger.info(f'  Have fragment perplexity: {ppl[0]}')

        x_ppl = entropy(
            observer_logits.to('cuda:0'),
            performer_logits.to('cuda:0'),
            encodings.to('cuda:0'),
            observer_model.tokenizer.pad_token_id
        )

        observer_model.logger.info(f'  Have fragment cross perplexity: {x_ppl[0]}')

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        observer_model.logger.info('  Binoculars score: %s',
                                    binoculars_scores[0])

    except RuntimeError as runtime_error:

        observer_model.logger.error(runtime_error)

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string = 'OOM'

        # Otherwise enter NAN:
        else:
            error_string = 'NAN'

        ppl = [error_string]
        x_ppl = [error_string]
        binoculars_scores = [error_string]

    # Record the results
    experiment.dependent_vars[
        'binoculars_score'].append(str(binoculars_scores[0]))

    experiment.dependent_vars[
        'perplexity'].append(str(ppl[0]))

    experiment.dependent_vars[
        'cross-perplexity'].append(str(x_ppl[0]))

    experiment.dependent_vars[
        'length_words'].append(fragment_length)

    experiment.dependent_vars[
        'length_tokens'].append(fragment_length_tokens)

    experiment.dependent_vars[
        'data_source'].append(record['Data source'])

    experiment.dependent_vars[
        'generating_model'].append(record['Generation model'])

    experiment.dependent_vars['author'].append(choice)
    experiment.dependent_vars['text'].append(text_fragment_string)
