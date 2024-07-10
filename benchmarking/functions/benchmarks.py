'''Collection of function to run benchmarks'''

from __future__ import annotations
from typing import List #, Callable

# import time
import random
import logging
# import tracemalloc
# from random import sample

# from multiprocessing import Process

# import torch
# import benchmarking.configuration as config
# import benchmarking.classes.llm as llm_class
# import benchmarking.classes.experiment as experiment_class
from benchmarking.functions.metrics import perplexity, entropy

# Comment ##############################################################
# Code ########################################################################

# def model_loading_benchmark(
#         experiment: Callable = None,
#         llm: Callable = None
# ) -> None:

#     '''Main benchmark function to time loading llm and tokenizer'''

#     # Since we are benchmarking the load time here - we need to clear
#     # then llm so we can reload it while timing. Not great, since doing
#     # it this way causes an extra load per batch, but this set-up is
#     # much better for the many other benchmarks that use this test
#     # harness
#     llm.clear()

#     # Time the loading of the model
#     loading_start_time = time.time()
#     llm.load()
#     total_load_time = time.time() - loading_start_time

#     # Record the results
#     experiment.dependent_vars['load_time'].append(total_load_time)


# def generation_rate_benchmark(
#         experiment: Callable = None,
#         llm: Callable = None
# ) -> None:

#     '''Main function to run generation rate benchmark'''

#     # Time the prompting of the model
#     inference_start = time.time()
#     _, output_ids = llm.prompt(config.PROMPT)
#     total_inference_time = time.time() - inference_start

#     # Count tokens generated
#     tokens_generated = len(output_ids[0])

#     # Calculate the generation rate
#     avg_generation_rate = tokens_generated / total_inference_time

#     # Record the results
#     experiment.dependent_vars['tokens_generated'].append(tokens_generated)
#     experiment.dependent_vars['inference_time'].append(total_inference_time)
#     experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


# def decoding_strategy_benchmark(
#         experiment: Callable = None,
#         llm: Callable = None
# ) -> None:

#     '''Main function to run decoding strategy benchmark'''

#     # Time the prompting of the model
#     inference_start = time.time()
#     _, output_ids = llm.prompt(config.PROMPT)
#     total_inference_time = time.time() - inference_start

#     # Count tokens generated
#     tokens_generated = len(output_ids[0])

#     # Calculate the generation rate
#     avg_generation_rate = tokens_generated / total_inference_time

#     # Record the results
#     experiment.dependent_vars['tokens_generated'].append(tokens_generated)
#     experiment.dependent_vars['inference_time'].append(total_inference_time)
#     experiment.dependent_vars['generation_rate'].append(avg_generation_rate)


# def encoding_memory_benchmark(
#         experiment: Callable = None,
#         llm: Callable = None
# ) -> None:

#     '''Main function to run encoding memory benchmark'''

#     # Sample the test text
#     text_list = config.ENCODING_TEST_TEXT.split(' ')

#     text_list_sample = sample(
#         text_list,
#         experiment.independent_vars['input_length'][-1]
#     )

#     input_text = ' '.join(text_list_sample)

#     # Reset memory stats for all devices
#     for device in config.AVAILABLE_GPUS:
#         torch.cuda.reset_peak_memory_stats(device = device)

#     # Time the encoding
#     encoding_start = time.time()

#     # Encode
#     encodings = llm.tokenizer(
#         input_text,
#         return_tensors = 'pt',
#         return_token_type_ids = False
#     ).to('cuda')

#     encoding_time = time.time() - encoding_start

#     # Get encoded fragment length
#     fragment_length = encodings['input_ids'].shape[1]

#     # Get encoding rate
#     encoding_rate=fragment_length / encoding_time

#     # Get total peak memory
#     peak_memory = 0

#     for device in config.AVAILABLE_GPUS:
#         peak_memory += torch.cuda.max_memory_allocated(device = device) / (10 ** 9)

#     # Record the results
#     experiment.dependent_vars['peak_memory'].append(peak_memory)
#     experiment.dependent_vars['tokens'].append(fragment_length)
#     experiment.dependent_vars['encoding_time'].append(encoding_time)
#     experiment.dependent_vars['encoding_rate'].append(encoding_rate)


# def logits_calculation_benchmark(
#         experiment: Callable = None,
#         llm: Callable = None
# ) -> None:

#     '''Main function to run logits cpu benchmark'''

#     # Sample the test text
#     text_list = config.ENCODING_TEST_TEXT.split(' ')

#     text_list_sample = sample(
#         text_list,
#         experiment.independent_vars['input_length'][-1]
#     )

#     input_text=' '.join(text_list_sample)

#     # Encode
#     encodings = llm.tokenizer(
#         input_text,
#         return_tensors = 'pt',
#         return_token_type_ids = False
#     )

#     # If this is not a CPU run, move encoding to GPU
#     if experiment.independent_vars['device_map'][-1] != 'cpu':
#         encodings = encodings.to('cuda')

#     # Get encoded fragment length
#     fragment_length = encodings['input_ids'].shape[1]

#     # Start memory tracking using the correct strategy based on device map
#     if experiment.independent_vars['device_map'][-1] != 'cpu':

#         # Reset memory stats for all GPUs
#         for device in config.AVAILABLE_GPUS:
#             torch.cuda.reset_peak_memory_stats(device = device)

#     elif experiment.independent_vars['device_map'][-1] == 'cpu':
#         tracemalloc.start()

#     # Time the logits calculation
#     logits_start = time.time()
#     _ = llm.model(**encodings).logits
#     logits_time = time.time() - logits_start

#     # Get calculation rate
#     rate=fragment_length / logits_time

#     # Get max memory using the correct strategy based on device map
#     if experiment.independent_vars['device_map'][-1] != 'cpu':
#         max_memory = 0

#         for device in config.AVAILABLE_GPUS:
#             device_max_memory = torch.cuda.max_memory_allocated(device=device)
#             device_max_memory = device_max_memory / (10 ** 9)
#             max_memory += device_max_memory

#     elif experiment.independent_vars['device_map'][-1] == 'cpu':

#         _, max_memory = tracemalloc.get_traced_memory()
#         max_memory = max_memory / (10 ** 6)
#         tracemalloc.stop()

#     # Record the results
#     experiment.dependent_vars['max_memory'].append(max_memory)
#     experiment.dependent_vars['tokens'].append(fragment_length)
#     experiment.dependent_vars['logits_time'].append(logits_time)
#     experiment.dependent_vars['rate'].append(rate)


def perplexity_ratio(
        run_dict: dict = None,
        llms: List[dict] = None,
        data = None
) -> None:

    '''Main function to run perplexity ratio score benchmark'''

    # Get the logger
    logger = logging.getLogger('benchmarking.perplexity_ratio')

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

    # Fence to catch CUDA OOM
    try:
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

        ppl = perplexity(encodings, writer_logits)

        x_ppl = entropy(
            reader_logits,
            writer_logits.to(reader_model.device_map),
            encodings,
            reader_model.tokenizer.pad_token_id
        )

        perplexity_ratio_scores = ppl / x_ppl
        perplexity_ratio_scores = perplexity_ratio_scores.tolist()

    except RuntimeError as runtime_error:

        logger.error(runtime_error)

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string = 'OOM'

        # Otherwise enter NAN:
        else:
            error_string = 'NAN'

        ppl = [error_string]
        x_ppl = [error_string]
        perplexity_ratio_scores = [error_string]

    # Record the results
    result = {'perplexity_ratio_score': str(perplexity_ratio_scores[0])}
    result['perplexity'] = str(ppl[0])
    result['cross-perplexity'] = str(x_ppl[0])
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
