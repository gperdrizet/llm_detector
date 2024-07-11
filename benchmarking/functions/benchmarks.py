'''Collection of function to run benchmarks'''

from __future__ import annotations
from typing import List

import time
import random
from random import sample

import torch
import benchmarking.configuration as config
import benchmarking.functions.helper as helper_funcs
from benchmarking.functions.metrics import perplexity, entropy

# Comment ##############################################################
# Code ########################################################################

def model_loading(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main benchmark function to time loading llm and tokenizer'''

    # Don't do anything with the data for this benchmark
    # it should contain the default 'None' value
    if data is None:
        pass

    # Re-assign model for clarity
    llm = llms[0]

    # Since we are benchmarking the load time here - we need to clear
    # then llm so we can reload it while timing. Not great, since doing
    # it this way causes an extra load per batch, but this set-up is
    # much better for the many other benchmarks that use this test
    # harness
    llm.clear()

    # Time the loading of the model
    loading_start_time = time.time()
    llm.load()
    load_time = time.time() - loading_start_time

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['cache_dir'] = run_dict['cache_dir']
    result['device_map'] = run_dict['device_map']
    result['cpu_cores'] = run_dict['cpu_cores']
    result['load_time'] = load_time

    return result


def generation(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run generation rate benchmark'''

    # Don't do anything with the data for this benchmark
    # it should contain the default 'None' value
    if data is None:
        pass

    # Re-assign model for clarity
    llm = llms[0]

    # Start memory tracking using the correct strategy based on device map
    helper_funcs.start_memory_tracking(device_map = run_dict['device_map'])

    # Time the prompting of the model
    inference_start = time.time()
    _, output_ids = llm.prompt(config.PROMPT)
    total_inference_time = time.time() - inference_start

    # Get peak memory using the correct strategy based on device map
    peak_memory = helper_funcs.get_peak_memory(device_map = run_dict['device_map'])

    # Count tokens generated
    output_length_tokens = len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate = output_length_tokens / total_inference_time

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['device_map'] = run_dict['device_map']
    result['quantization'] = run_dict['quantization']
    result['cpu_cores'] = run_dict['cpu_cores']
    result['max_new_tokens'] = run_dict['max_new_tokens']
    result['output_length_tokens'] = output_length_tokens
    result['inference_time'] = total_inference_time
    result['generation_rate'] = avg_generation_rate
    result['peak_memory'] = peak_memory

    return result


def decoding_strategy(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run decoding strategy benchmark'''

    # Don't do anything with the data for this benchmark
    # it should contain the default 'None' value
    if data is None:
        pass

    # Re-assign model for clarity
    llm = llms[0]

    # Start memory tracking using the correct strategy based on device map
    helper_funcs.start_memory_tracking(device_map = run_dict['device_map'])

    # Time the prompting of the model
    inference_start = time.time()
    _, output_ids = llm.prompt(config.PROMPT)
    total_inference_time = time.time() - inference_start

    # Get peak memory using the correct strategy based on device map
    peak_memory = helper_funcs.get_peak_memory(device_map = run_dict['device_map'])

    # Count tokens generated
    output_length_tokens = len(output_ids[0])

    # Calculate the generation rate
    avg_generation_rate = output_length_tokens / total_inference_time

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['device_map'] = run_dict['device_map']
    result['max_new_tokens'] = run_dict['max_new_tokens']
    result['decoding_strategy'] = run_dict['decoding_strategy']
    result['output_length_tokens'] = output_length_tokens
    result['inference_time'] = total_inference_time
    result['generation_rate'] = avg_generation_rate
    result['peak_memory'] = peak_memory

    return result


def string_encoding(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run encoding memory benchmark'''

    # Don't do anything with the data for this benchmark
    # it should contain the default 'None' value
    if data is None:
        pass

    # Re-assign model for clarity
    llm = llms[0]

    # Sample the test text
    text_list = config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample = sample(
        text_list,
        run_dict['input_length_words']
    )

    input_text = ' '.join(text_list_sample)

    # Start memory tracking using the correct strategy based on device map
    helper_funcs.start_memory_tracking(device_map = run_dict['device_map'])

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
    output_length_tokens = encodings['input_ids'].shape[1]

    # Get encoding rate
    encoding_rate = output_length_tokens / encoding_time

    # Get peak memory using the correct strategy based on device map
    peak_memory = helper_funcs.get_peak_memory(device_map = run_dict['device_map'])

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['device_map'] = run_dict['device_map']
    result['input_length_words'] = run_dict['input_length_words']
    result['peak_memory'] = peak_memory
    result['output_length_tokens'] = output_length_tokens
    result['encoding_time'] = encoding_time
    result['encoding_rate'] = encoding_rate

    return result


def logits_calculation(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run logits cpu benchmark'''

    # Don't do anything with the data for this benchmark
    # it should contain the default 'None' value
    if data is None:
        pass

    # Re-assign model for clarity
    llm = llms[0]

    # Generate a sample text fragment from the test text
    text_list = config.ENCODING_TEST_TEXT.split(' ')

    text_list_sample = sample(
        text_list,
        run_dict['input_length_words']
    )

    input_text=' '.join(text_list_sample)

    # Encode the text sample
    encodings = llm.tokenizer(
        input_text,
        return_tensors = 'pt',
        return_token_type_ids = False
    )

    # If this is not a CPU run, move encoding to GPU
    if run_dict['device_map'] != 'cpu':
        encodings = encodings.to('cuda')

    # Get encoded fragment length
    output_length_tokens = encodings['input_ids'].shape[1]

    # Start memory tracking using the correct strategy based on device map
    helper_funcs.start_memory_tracking(device_map = run_dict['device_map'])

    # Time the logits calculation
    logits_start = time.time()
    _ = llm.model(**encodings).logits
    calculation_time = time.time() - logits_start

    # Get calculation rate
    calculation_rate = output_length_tokens / calculation_time

    # Get peak memory using the correct strategy based on device map
    peak_memory = helper_funcs.get_peak_memory(device_map = run_dict['device_map'])

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['device_map'] = run_dict['device_map']
    result['quantization'] = run_dict['quantization']
    result['input_length_words'] = run_dict['input_length_words']
    result['peak_memory'] = peak_memory
    result['output_length_tokens'] = output_length_tokens
    result['calculation_time'] = calculation_time
    result['calculation_rate'] = calculation_rate

    return result


def perplexity_ratio(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run perplexity ratio score benchmark'''

    # Don't do anything with the run dict for this benchmark
    # it should contain the default 'None' value
    if run_dict is None:
        pass

    # Set the models to evaluation mode to deactivate any dropout
    # modules the is done to ensure reproducibility of results during
    # evaluation
    for i, _ in enumerate(llms):
        llms[i].model.eval()

    # Add end of sequence to the reader's tokenizer for the pad
    # token if not defined
    if not llms[0].tokenizer.pad_token:
        llms[0].tokenizer.pad_token = llms[0].tokenizer.eos_token

    # Assign the models to human readable names for clarity
    reader_model = llms[0]
    writer_model = llms[1]

    # Find out how many records we have for use later
    num_records = len(data)

    # Sample the data, repeating the sampling until
    # we get a valid text fragment
    text_fragment_string = None

    while text_fragment_string is None:

        # Pick a random record number
        record_num = random.randint(0, num_records - 1)

        # Pull the record and get the human and synthetic texts
        record = data[record_num]
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
        text_fragment_string = ' '.join(text_fragment_list)

    # Now calculate the perplexity ratio score

    # Encode
    encodings = reader_model.tokenizer(
        text_fragment_string,
        return_tensors = 'pt',
        return_token_type_ids = False
    ).to(reader_model.device_map)

    # Get input ids as list for logging/data collection
    fragment_length_tokens = encodings['input_ids'].shape[1]

    # Calculate logits
    reader_logits = reader_model.model(**encodings).logits
    writer_logits = writer_model.model(**encodings).logits

    # Calculate perplexity ratio score
    ppl = perplexity(encodings, writer_logits)

    x_ppl = entropy(
        reader_logits,
        writer_logits.to(reader_model.device_map),
        encodings,
        reader_model.tokenizer.pad_token_id
    )

    perplexity_ratio_scores = ppl / x_ppl
    perplexity_ratio_scores = perplexity_ratio_scores.tolist()

    # Record the results
    result = {'iteration': run_dict['iteration']}
    result['hf_model_string'] = run_dict['hf_model_string']
    result['device_map'] = run_dict['device_map']
    result['perplexity_ratio_score'] = float(perplexity_ratio_scores[0])
    result['perplexity'] = float(ppl[0])
    result['cross-perplexity'] = float(x_ppl[0])
    result['length_words'] = fragment_length
    result['length_tokens'] = fragment_length_tokens
    result['data_source'] = record['Data source']
    result['generating_model'] = record['Generation model']
    result['reader_model'] = reader_model.hf_model_string
    result['writer_model'] = writer_model.hf_model_string
    result['reader_device'] = reader_model.device_map
    result['writer_device'] = writer_model.device_map
    result['author'] = choice
    result['text'] = text_fragment_string

    return result
