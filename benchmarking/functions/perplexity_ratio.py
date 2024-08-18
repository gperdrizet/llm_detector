'''Functions for detecting LLM generated text. Perplexity ratio score
metric inspired by: https://arxiv.org/abs/2401.12070'''

from __future__ import annotations
from typing import Callable

import json
import random
import torch
import benchmarking.configuration as config
import benchmarking.functions.helper as helper_funcs
from benchmarking.functions.metrics import perplexity, entropy
import benchmarking.classes.llm as llm_class

# Comment ##############################################################
# Code ########################################################################

def perplexity_ratio_score():
    '''Computes perplexity ratio score on test string.'''

    # Start logger
    logger = helper_funcs.start_logger('hans_data_perplexity_ratio_score')
    logger.info('Starting binoculars')

    # Set reuseable devices - using both chips on a single K80 for now
    reader_device = 'cuda:1'
    writer_device = 'cuda:2'

    # Set available CPU cores
    torch.set_num_threads(4)

    # Instantiate two instances of the model, one base for the reader
    # and one instruct for the writer. Use different GPUs.
    reader_model = llm_class.Llm(
        hf_model_string = config.READER_MODEL,
        device_map = reader_device
    )

    writer_model = llm_class.Llm(
        hf_model_string = config.WRITER_MODEL,
        device_map = writer_device
    )

    # Load the models
    reader_model.load()
    writer_model.load()

    # Set the models to evaluation mode to deactivate any dropout
    # modules to ensure reproducibility of results during evaluation
    reader_model.model.eval()
    writer_model.model.eval()

    logger.info('Models loaded')

    # Add end of sequence for the pad token if one has not been defined
    if not reader_model.tokenizer.pad_token:
        reader_model.tokenizer.pad_token = reader_model.tokenizer.eos_token

    logger.info('Tokenizer prepared')

    ###############################################################
    # Next we are going to loop over all of the core data files   #
    # provided in the perplexity ratio repo and compute the       #
    # perplexity, cross-perplexity and perplexity ratio score on  #
    # all of if so we can make some plots and get a better feel   #
    # for the distribution.                                       #
    ###############################################################

    # Collector dict for results
    results = {
        'Fragment': [],
        'Fragment length (tokens)': [],
        'Dataset': [],
        'Source': [],
        'String': [],
        'Reader peak memory (GB)': [],
        'Writer peak memory (GB)': [],
        'Perplexity': [],
        'Cross-perplexity': [],
        'Perplexity ratio score': [],
    }

    # The hans datasets use different keys for the human text, use this
    # dict. to look up the correct one based on the data source
    human_text_keys = {
        'pubmed-falcon7': 'article',
        'pubmed-llama2-13': 'article',
        'cnn-falcon7': 'article',
        'cnn-llama2-13': 'article',
        'cc_news-falcon7': 'text',
        'cc_news-llama2-13': 'text'
    }

    # Output file
    results_datafile = f'{config.HANS_DATA_PATH}/{config.PERPLEXITY_OUTPUT_FILE_NAME}'

    # Counter for total text fragments scored
    fragment_count = 0

    # Loop on JSON lines...
    for dataset, dataset_file in config.HANS_DATA_FILES.items():

        # Get the correct key for the human text in this dataset
        human_key = human_text_keys[dataset]

        with open(dataset_file, encoding = 'utf-8') as f:
            for line in f:
                record = json.loads(line)

                # Only consider records which have text, grab the human
                # and synthetic for further processing
                texts = {}

                if human_key in list(record.keys()):
                    texts['human'] = record[human_key]
                    texts['synthetic'] = record[list(record.keys())[-1]]

                    # Now we have both the human and synthetic texts
                    # for this record. Next thing to do is break them
                    # into smaller, randomly sized chunks for encoding
                    # and perplexity ratio score calculation. Randomly
                    # sized so that we can look at length as a factor
                    # later
                    for source, text_string in texts.items():

                        # Loop to break the text into randomly sized chunks

                        # Split text to list
                        text_list = text_string.split(' ')

                        # Get the total length
                        total_length = len(text_list)

                        # Counters
                        i,j = 0,0

                        # How long is it?
                        logger.info("Text length: %s", total_length)

                        # Loop until the right edge is past the end
                        while j < total_length:

                            # Pick a random length between 50 and 500
                            # tokens
                            slice_length = random.randint(50, 300)

                            # If the slice length is greater than the
                            # length of the input tokens, use all of
                            # them
                            if slice_length > total_length:
                                slice_length = total_length

                            # Set the window
                            j = i + slice_length

                            # Grab the slice
                            text_list_slice = text_list[i:j]

                            # Make it a string
                            text_string_slice = ' '.join(text_list_slice)

                            # Run guts of loop
                            results = main_loop(
                                results = results,
                                text_string_slice = text_string_slice,
                                fragment_count = fragment_count,
                                dataset = dataset,
                                source = source,
                                reader_model = reader_model,
                                writer_model = writer_model,
                                reader_device = reader_device,
                                writer_device = writer_device,
                                logger = logger
                            )

                            # Serialize collected results to JSON every 10
                            # loops
                            if fragment_count % 10 == 0:
                                with open(results_datafile, 'w',
                                          encoding = 'utf-8') as output:
                                    json.dump(results, output)

                            # Reset for the next loop
                            i = j
                            fragment_count += 1


def main_loop(
    results: dict = None,
    text_string_slice: str = None,
    fragment_count: int = None,
    dataset: str = None,
    source: str = None,
    reader_model: Callable = None,
    writer_model: Callable = None,
    reader_device: str = 'cuda:1',
    writer_device: str = 'cuda:2',
    logger: Callable = None
) -> None:

    '''Function to encapsulate GPU operations for
    calculation of perplexity ratio score on text fragment.'''

    # Reset peak memory so we can track this iterations allocation
    torch.cuda.reset_peak_memory_stats(device = reader_device)
    torch.cuda.reset_peak_memory_stats(device = writer_device)

    # Fence to catch CUDA OOM
    try:

        result = calculate_perplexity_ratio_score(
            text_string_slice = text_string_slice,
            reader_model = reader_model,
            writer_model = writer_model,
            reader_device = reader_device,
            logger = logger
        )

        (fragment_length, ppl, x_ppl, perplexity_ratio_scores) = result

    except RuntimeError as runtime_error:

        logger.error(runtime_error)

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string = 'OOM'

        # Otherwise enter NAN:
        else:
            error_string = 'NAN'

        fragment_length = error_string
        ppl = [error_string]
        x_ppl = [error_string]
        perplexity_ratio_scores = [error_string]

    # Get peak memory use for reader and writer
    writer_peak_memory = torch.cuda.max_memory_allocated(device = writer_device)
    reader_peak_memory = torch.cuda.max_memory_allocated(device = reader_device)
    writer_peak_memory = writer_peak_memory / (10 ** 9)
    reader_peak_memory = reader_peak_memory / (10 ** 9)

    results['Fragment'].append(str(fragment_count))
    results['Fragment length (tokens)'].append(str(fragment_length))
    results['Dataset'].append(str(dataset))
    results['Source'].append(str(source))
    results['String'].append(text_string_slice)
    results['Reader peak memory (GB)'].append(str(reader_peak_memory))
    results['Writer peak memory (GB)'].append(str(writer_peak_memory))
    results['Perplexity'].append(str(ppl[0]))
    results['Cross-perplexity'].append(str(x_ppl[0]))
    results['Perplexity ratio score'].append(str(perplexity_ratio_scores[0]))

    print(f'Fragment: {fragment_count}')
    print(f'Fragment length (tokens): {fragment_length}')
    print(f'Dataset: {dataset}')
    print(f'Source: {source}')
    print(f'Text: {text_string_slice}')
    print(f'Reader peak memory (GB): {round(reader_peak_memory, 1)}')
    print(f'Writer peak memory (GB): {round(writer_peak_memory,1)}')
    print(f'Perplexity: {ppl[0]}')
    print(f'Cross-perplexity: {x_ppl[0]}')
    print(f'Binoculars score: {perplexity_ratio_scores[0]}')
    print()

    # Put the models and results dict back in the queue
    return results

def calculate_perplexity_ratio_score(
    text_string_slice: str = None,
    reader_model: Callable = None,
    writer_model: Callable = None,
    reader_device: str = 'cuda:1',
    logger: Callable = None
):

    '''Computes the perplexity ratio score'''

    # Encode
    encodings = reader_model.tokenizer(
        text_string_slice,
        return_tensors = 'pt',
        return_token_type_ids = False
    ).to(reader_device)

    # Get input ids only as list for later logging/data collection
    fragment_length = encodings['input_ids'].shape[1]

    reader_logits = reader_model.model(**encodings).logits
    writer_logits = writer_model.model(**encodings).logits

    logger.info('Slice encoded')
    logger.info('Slice length: %s', encodings["input_ids"].shape[1])
    logger.info('Logits length: %s', {writer_logits.shape})

    ppl = perplexity(encodings, writer_logits)
    logger.info('Have slice perplexity')

    x_ppl = entropy(
        reader_logits.to('cuda:0'),
        writer_logits.to('cuda:0'),
        encodings.to('cuda:0'),
        reader_model.tokenizer.pad_token_id
    )

    logger.info('Have cross perplexity')

    perplexity_ratio_scores = ppl / x_ppl
    perplexity_ratio_scores = perplexity_ratio_scores.tolist()
    logger.info('Perplexity ratio score: %s', perplexity_ratio_scores[0])

    return fragment_length, ppl, x_ppl, perplexity_ratio_scores
