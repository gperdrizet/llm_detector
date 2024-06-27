'''Functions for detecting LLM generated text. Binoculars score
metric inspired by: https://arxiv.org/abs/2401.12070'''

from __future__ import annotations
from typing import Callable

import json
import random
import torch
import configuration as config
import functions.helper as helper_funcs
from functions.metrics import perplexity, entropy
import classes.llm as llm_class

def binoculars():
    '''Computes binoculars score on test string.'''

    # Start logger
    logger=helper_funcs.start_logger()
    logger.info('Starting binoculars')

    # Set reuseable devices - using both chips on a single K80 for now
    observer_device='cuda:1'
    performer_device='cuda:2'

    # Set available CPU cores - doing this from the LLM class does not seem to work
    torch.set_num_threads(16)

    # Instantiate two instances of the model, one base for the observer
    # and one instruct for the performer. Use different GPUs.
    observer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B',
        device_map=observer_device,
        logger=logger
    )

    performer_model=llm_class.Llm(
        hf_model_string='meta-llama/Meta-Llama-3-8B-instruct',
        device_map=performer_device,
        logger=logger
    )

    # Load the models
    observer_model.load()
    performer_model.load()

    # Set the models to evaluation mode to deactivate any dropout modules
    # the is done to ensure reproducibility of results during evaluation
    observer_model.model.eval()
    performer_model.model.eval()

    logger.info('Models loaded')

    # Add end of sequence for the pad token if one has not been defined
    if not observer_model.tokenizer.pad_token:
        observer_model.tokenizer.pad_token=observer_model.tokenizer.eos_token

    logger.info('Tokenizer prepared')

    ###############################################################
    # Next we are going to loop over all of the core data files   #
    # provided in the binoculars repo and compute the perplexity, #
    # cross-perplexity and binocular score on all of if so we can #
    # make some plots and get a better feel for the distribution  #
    ###############################################################

    # Collector dict for results
    results={
        'Fragment': [],
        'Fragment length (tokens)': [],
        'Dataset': [],
        'Source': [],
        'String': [],
        'Observer peak memory (GB)': [],
        'Performer peak memory (GB)': [],
        'Perplexity': [],
        'Cross-perplexity': [],
        'Binoculars score': [],
    }

    # Output file
    results_datafile=f'{config.BINOCULARS_DATA_PATH}/scores.json'

    # Counter for total text fragments scored
    fragment_count=0

    # Loop on JSON lines...
    for dataset, dataset_file in config.BINOCULARS_DATA_FILES.items():
        with open(dataset_file, encoding='utf-8') as f:
            for line in f:
                record=json.loads(line)

                # Only consider records which have text, grab the human
                # and synthetic for further processing
                texts={}

                if 'text' in list(record.keys()):
                    texts['human']=record['text']
                    texts['synthetic']=record[list(record.keys())[-1]]

                    # Now we have both the human and synthetic texts for this record.
                    # Next thing to do is break them into smaller, randomly sized chunks
                    # for encoding and binoculars score calculation. Randomly sized so
                    # that we can look at length as a factor later.
                    for source, text_string in texts.items():

                        # Loop to break the text into randomly sized chunks

                        # Split text to list
                        text_list=text_string.split(' ')

                        # Get the total length
                        total_length=len(text_list)

                        # Counters
                        i,j=0,0

                        # How long is it?
                        logger.info("Text length: %s", total_length)

                        # Loop until the right edge is past the end
                        while j < total_length:

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

                            # Run guts of loop
                            results=main_loop(
                                results=results,
                                text_string_slice=text_string_slice,
                                fragment_count=fragment_count,
                                dataset=dataset,
                                source=source,
                                observer_model=observer_model,
                                performer_model=performer_model,
                                observer_device=observer_device,
                                performer_device=performer_device,
                                logger=logger
                            )

                            # Serialize collected results to JSON every 10 loops
                            if fragment_count % 10 == 0:
                                with open(results_datafile, 'w', encoding='utf-8') as output:
                                    json.dump(results, output)

                            # Reset for the next loop
                            i=j
                            fragment_count += 1


def main_loop(
    results: dict=None,
    text_string_slice: str=None,
    fragment_count: int=None,
    dataset: str=None,
    source: str=None,
    observer_model: Callable=None,
    performer_model: Callable=None,
    observer_device: str='cuda:1',
    performer_device: str='cuda:2',
    logger: Callable=None
) -> None:

    '''Function to encapsulate GPU operations for
    calculation of binoculars score on text fragment.'''

    # Reset peak memory so we can track this iterations allocation
    torch.cuda.reset_peak_memory_stats(device=observer_device)
    torch.cuda.reset_peak_memory_stats(device=performer_device)

    # Fence to catch CUDA OOM
    try:

        fragment_length, ppl, x_ppl, binoculars_scores=calculate_binoculars_score(
            text_string_slice=text_string_slice,
            observer_model=observer_model,
            performer_model=performer_model,
            observer_device=observer_device,
            logger=logger
        )

    except RuntimeError as runtime_error:

        logger.error(runtime_error)

        # For out of memory enter OOM
        if 'CUDA out of memory' in str(runtime_error):
            error_string='OOM'

        # Otherwise enter NAN:
        else:
            error_string='NAN'

        fragment_length=error_string
        ppl=[error_string]
        x_ppl=[error_string]
        binoculars_scores=[error_string]

    # Get peak memory use for observer and performer
    performer_peak_memory=torch.cuda.max_memory_allocated(device=performer_device) / (10 ** 9)
    observer_peak_memory=torch.cuda.max_memory_allocated(device=observer_device) / (10 ** 9)

    results['Fragment'].append(str(fragment_count))
    results['Fragment length (tokens)'].append(str(fragment_length))
    results['Dataset'].append(str(dataset))
    results['Source'].append(str(source))
    results['String'].append(text_string_slice)
    results['Observer peak memory (GB)'].append(str(observer_peak_memory))
    results['Performer peak memory (GB)'].append(str(performer_peak_memory))
    results['Perplexity'].append(str(ppl[0]))
    results['Cross-perplexity'].append(str(x_ppl[0]))
    results['Binoculars score'].append(str(binoculars_scores[0]))

    print(f'Fragment: {fragment_count}')
    print(f'Fragment length (tokens): {fragment_length}')
    print(f'Dataset: {dataset}')
    print(f'Source: {source}')
    print(f'Text: {text_string_slice}')
    print(f'Observer peak memory (GB): {round(observer_peak_memory, 1)}')
    print(f'Performer peak memory (GB): {round(performer_peak_memory,1)}')
    print(f'Perplexity: {ppl[0]}')
    print(f'Cross-perplexity: {x_ppl[0]}')
    print(f'Binoculars score: {binoculars_scores[0]}')
    print()

    # Put the models and results dict back in the queue
    return results

def calculate_binoculars_score(
    text_string_slice: str=None,
    observer_model: Callable=None,
    performer_model: Callable=None,
    observer_device: str='cuda:1',
    logger: Callable=None
):

    '''Computes the binoculars score'''

    # Encode
    encodings=observer_model.tokenizer(
        text_string_slice,
        return_tensors="pt",
        return_token_type_ids=False
    ).to(observer_device)

    # Get input ids only as list for later logging/data collection
    fragment_length=encodings['input_ids'].shape[1]

    observer_logits=observer_model.model(**encodings).logits
    performer_logits=performer_model.model(**encodings).logits

    logger.info('Slice encoded')
    logger.info('Slice length: %s', encodings["input_ids"].shape[1])
    logger.info('Logits length: %s', {performer_logits.shape})

    ppl=perplexity(encodings, performer_logits)
    logger.info('Have slice perplexity')

    x_ppl=entropy(
        observer_logits.to('cuda:0'),
        performer_logits.to('cuda:0'),
        encodings.to('cuda:0'),
        observer_model.tokenizer.pad_token_id
    )

    logger.info('Have cross perplexity')

    binoculars_scores = ppl / x_ppl
    binoculars_scores = binoculars_scores.tolist()
    logger.info('Binoculars score: %s', binoculars_scores[0])

    return fragment_length, ppl, x_ppl, binoculars_scores
