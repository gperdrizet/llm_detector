'''Collection of functions for data manipulation, parsing and exploration. Includes
functions to download, parse, combine and semantically split text data sets for 
perplexity ratio scoring.'''

from __future__ import annotations
from typing import Tuple

# Standard library imports
import os
import glob
import csv
import json
import zipfile
import urllib.request
import logging
import multiprocessing as mp
from pathlib import Path
from itertools import product
from statistics import mean, stdev

# PyPI imports
import kaggle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset, utils
from semantic_text_splitter import TextSplitter # pylint: disable=no-name-in-module
from tokenizers import Tokenizer

# Internal imports
import functions.helper as helper_funcs # pylint: disable=import-error
import functions.multiprocess_logging as log_funcs # pylint: disable=import-error
import configuration as config # pylint: disable=import-error

def parse_hans_data(
    hans_datasets: dict = None,
    hans_data: dict = None,
    hans_metadata: dict = None,
    binoculars_data_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Parses and collects datasets from Hans et al (2024) Binocular publication. 
    Takes dict describing datasets with names and file paths and two empty dicts 
    to hold output: one for metadata and one for the collected datasets. Returns 
    parses and returns populated data and metadata pandas dataframes.'''

    # The hans datasets use different keys for the human text, use this
    # dict. to look up the correct one based on the data source
    human_text_keys = {
        'cc_news': 'text',
        'cnn': 'article',
        'pubmed': 'article'
    }

    # Loop on outer level of hand datasets dict.
    for generation_model, datasets in hans_datasets.items():

        # loop on each inner level of hans datasets dict
        for data_source, datafile_name in datasets.items():

            # Build the absolute file name for this dataset
            dataset_file = f'{binoculars_data_path}/{datafile_name}'

            # Set initial values for some collector vars
            record_count = 1
            human_text_lengths = []
            synthetic_text_lengths = []
            human_text_fractions = []

            # Open the data file and loop on lines, loading each as JSON
            with open(dataset_file, encoding = 'utf-8') as f:
                for line in f:
                    record = json.loads(line)

                    # If we can find a text record in the JSON object, continue processing
                    if human_text_keys[data_source] in list(record.keys()):

                        # Get the human and synthetic text
                        human_text = record[human_text_keys[data_source]]
                        synthetic_text = record[list(record.keys())[-1]]

                        # Get the texts' lengths
                        human_text_length = len(human_text.split(' '))
                        synthetic_text_length = len(synthetic_text.split(' '))

                        # Collect the texts' lengths
                        human_text_lengths.append(human_text_length)
                        synthetic_text_lengths.append(synthetic_text_length)

                        # Get and collect the fraction of this record's text that is human
                        total_text_length = human_text_length + synthetic_text_length
                        human_text_fraction = human_text_length / total_text_length
                        human_text_fractions.append(human_text_fraction)

                        # Count this record
                        record_count += 1

                        # Add data from this record to the collected data result
                        hans_data['Generation model'].append(generation_model)
                        hans_data['Data source'].append(data_source)
                        hans_data['Human text length (words)'].append(human_text_length)
                        hans_data['Human text'].append(human_text)
                        hans_data['Synthetic text length (words)'].append(synthetic_text_length)
                        hans_data['Synthetic text'].append(synthetic_text)
                        hans_data['Human text fraction'].append(human_text_fraction)

            print(f'Parsed {generation_model}, {data_source} data: {record_count} records')

            # Add metadata from this dataset to the result
            hans_metadata['Generation model'].append(generation_model)
            hans_metadata['Data source'].append(data_source)
            hans_metadata['Records'].append(record_count)
            hans_metadata['Mean human text length (words)'].append(mean(human_text_lengths))
            hans_metadata['Human text length STD'].append(stdev(human_text_lengths))
            hans_metadata['Mean synthetic text length (words)'].append(mean(synthetic_text_lengths))
            hans_metadata['Synthetic text length STD'].append(stdev(synthetic_text_lengths))
            hans_metadata['Mean human text fraction'].append(mean(human_text_fractions))

    hans_data_df = pd.DataFrame.from_dict(hans_data)
    hans_metadata_df = pd.DataFrame.from_dict(hans_metadata)

    return hans_metadata_df, hans_data_df


def tf_idf(data_df: pd.DataFrame=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Calculates TF-IDF for human and synthetic texts from Hans data'''

    # Get lists of human and synthetic text
    human_texts=list(data_df['Human text'])
    synthetic_texts=list(data_df['Synthetic text'])

    # Set-up sklearn TFIDF vectorizer
    tfidf_vectorizer=TfidfVectorizer(input='content')

    # Get human and synthetic TFIDF values and convert to dataframe
    human_tfidf_vector=tfidf_vectorizer.fit_transform(human_texts)

    human_tfidf_df=pd.DataFrame(
        human_tfidf_vector.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    synthetic_tfidf_vector=tfidf_vectorizer.fit_transform(synthetic_texts)

    synthetic_tfidf_df=pd.DataFrame(
        synthetic_tfidf_vector.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    return human_tfidf_df, synthetic_tfidf_df



def get_data() -> None:
    '''Main function to run steps in data acquisition pipeline.'''

    logger=helper_funcs.start_logger(
        logfile_name=f'{config.LOG_PATH}/data_acquisition.log',
        logger_name=f'{__name__}.get_data'
    )

    # Download the data
    logger.info('Starting data download')
    download_raw_data()
    logger.info('Data download complete')

    # Parse the data
    logger.info('Starting data parse')
    parsed_text=parse_raw_data()
    logger.info('Data parse complete')

    # Save the parsed text
    logger.info('Saving parsed text')
    save_parsed_text(parsed_text)
    logger.info('Finished')


def download_raw_data() -> None:
    '''Downloads raw data from internet sources'''

    # Get the logger
    function_logger=logging.getLogger(f'{__name__}.download_raw_data')

    ########
    # Hans #
    ########

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/hans'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Data source info
    hans_generating_models=['falcon7','llama2_13']
    hans_data_sources=['cnn','cc_news','pubmed']
    hans_base_url=('https://raw.githubusercontent.com'+
        '/ahans30/Binoculars/refs/heads/main/datasets/core')

    # Loop on generating models and data sources, downloading files for each
    for generating_model, data_source in product(hans_generating_models, hans_data_sources):
        output_file=f'{output_directory}/{generating_model}-{data_source}.jsonl'

        # Only download the file if we don't already have it
        if Path(output_file).is_file() is False:
            data_url=f'{hans_base_url}/{data_source}/{data_source}-{generating_model}.jsonl'
            _=urllib.request.urlretrieve(data_url, output_file)

            function_logger.info(
                'Finished downloading Hans %s-%s data',
                data_source,
                generating_model
            )

        else:
            function_logger.info(
                'Already have Hans %s-%s data',
                data_source,
                generating_model
            )


    ##########
    # Gerami #
    ##########

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/gerami'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Output file
    output_file=f'{output_directory}/ai-vs-human-text.zip'

    # Only download the file if we don't already have it
    if Path(output_file).is_file() is False:
        kaggle.api.dataset_download_files('shanegerami/ai-vs-human-text', path=output_directory)

        # Unzip the data
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_directory)

        function_logger.info('Finished downloading Gerami data')

    else:
        function_logger.info('Already have Gerami data')

    ############
    # Grinberg #
    ############

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/grinberg'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Output file
    output_file=f'{output_directory}/human-vs-llm-text-corpus.zip'

    # Only download the file if we don't already have it
    if Path(output_file).is_file() is False:
        kaggle.api.dataset_download_files(
            'starblasters8/human-vs-llm-text-corpus',
            path=output_directory
        )

        # Unzip the data
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_directory)

        function_logger.info('Finished downloading Grinberg data')

    else:
        function_logger.info('Already have Grinberg data')

    ##########
    # Gaggar #
    ##########

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/gaggar'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # File IO locations
    data_url=('https://github.com/HarshOza36/Detection-Of-Machine-Generated-Text'+
        '/raw/refs/heads/master/data/Final%20Dataset.zip')

    output_file=f'{output_directory}/data.zip'

    # Only download the file if we don't already have it
    if Path(output_file).is_file() is False:
        _=urllib.request.urlretrieve(data_url, output_file)

        # Unzip the data
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_directory)

        function_logger.info('Finished downloading Gaggar data')

    else:
        function_logger.info('Already have Gaggar data')

    ############
    # Yatsenko #
    ############

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/yatsenko'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Output directory for the data
    output_file=f'{output_directory}/data'

    # Only download the file if we don't already have it
    if Path(output_file).is_dir() is False:
        utils.disable_progress_bar()
        ds=load_dataset('artem9k/ai-text-detection-pile')

        # Save the dataset to disk
        ds.save_to_disk(output_file)

        function_logger.info('Finished downloading Yatsenko data')

    else:
        function_logger.info('Already have Yatsenko data')


def parse_raw_data():
    '''Parses and combines text from each data set'''

    # Get the logger
    function_logger=logging.getLogger(f'{__name__}.parse_raw_data')

    # Holder for results
    parsed_text={
        'Text': [],
        'Synthetic': [],
        'Author': [],
        'Source': []
    }

    ########
    # Hans #
    ########

    # Hans data set info
    hans_generating_models=['falcon7','llama2_13']
    hans_data_sources=['cnn','cc_news','pubmed']

    # Counters
    human_texts=0
    synthetic_texts=0

    # Loop on the generating model and original text source
    for generating_model, data_source in product(hans_generating_models, hans_data_sources):

        # Get the file path
        file_path=f'{config.RAW_DATA_PATH}/hans/{generating_model}-{data_source}.jsonl'

        # Loop on the JSON lines in the file, parsing each one
        with open(file_path, encoding='UTF-8') as input_file:
            for line in input_file:
                data=json.loads(line)

                # Get the generated text and add to parsed text
                parsed_text['Source'].append('Hans')
                parsed_text['Synthetic'].append(1)

                if generating_model == 'llama2_13':
                    parsed_text['Author'].append('LLaMA2-13B')
                    text=data['meta-llama-Llama-2-13b-hf_generated_text_wo_prompt']

                elif generating_model == 'falcon7':
                    parsed_text['Author'].append('Falcon-7B')
                    text=data['-fs-cml-models-Falcon-falcon-7b_generated_text_wo_prompt']

                parsed_text['Text'].append(text)

                synthetic_texts+=1

                # Get the human text and add to parsed text
                parsed_text['Source'].append('Hans')
                parsed_text['Synthetic'].append(0)
                parsed_text['Author'].append('Human')

                if 'article' in data.keys():
                    text=data['article']

                elif 'text' in data.keys():
                    text=data['text']

                parsed_text['Text'].append(text)

                human_texts+=1

    function_logger.info(f'Parsed Hans data: {human_texts + synthetic_texts} '+
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    ##########
    # Gerami #
    ##########

    # Data file path
    file_path=f'{config.RAW_DATA_PATH}/gerami/AI_Human.csv'

    # Counters
    human_texts=0
    synthetic_texts=0

    # Read the file
    with open(file_path, mode='r', encoding='UTF-8') as input_file:
        reader=csv.reader(input_file)

        # Loop on CSV rows, parsing each
        for i, row in enumerate(reader):

            # Skip the header row
            if i > 0:
                parsed_text['Source'].append('Gerami')

                if row[1] == '0.0':
                    parsed_text['Synthetic'].append(0)
                    parsed_text['Author'].append('Human')
                    human_texts+=1

                if row[1] == '1.0':
                    parsed_text['Synthetic'].append(1)
                    parsed_text['Author'].append('Unknown model')
                    synthetic_texts+=1

                parsed_text['Text'].append(row[0])

    function_logger.info(f'Parsed Gerami data: {human_texts + synthetic_texts} '+
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    ############
    # Grinberg #
    ############

    # Data file path
    file_path=f'{config.RAW_DATA_PATH}/grinberg/data.parquet'

    # Counters
    human_texts=0
    synthetic_texts=0

    # Read the file into a Pandas dataframe
    data_df=pd.read_parquet(file_path)
    data_df.head()

    # Extract texts and sources
    texts=data_df['text'].to_list()
    sources=data_df['source'].to_list()

    # Loop on text and source lists, parse and add the to results
    for text, source in zip(texts, sources):
        parsed_text['Source'].append('Grinberg')

        if source == 'Human':
            parsed_text['Synthetic'].append(0)
            parsed_text['Author'].append('Human')
            human_texts+=1

        if source != 'Human':
            parsed_text['Synthetic'].append(1)
            parsed_text['Author'].append('Unknown model')
            synthetic_texts+=1

        parsed_text['Text'].append(text)

    function_logger.info(f'Parsed Grinberg data: {human_texts + synthetic_texts} '+
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    ##########
    # Gagger #
    ##########

    # Data file path
    file_path=f'{config.RAW_DATA_PATH}/gaggar/Complete Dataset/FinalDataset.csv'

    # Counters
    human_texts=0
    synthetic_texts=0

    # Read the file
    with open(file_path, mode='r', encoding='UTF-8') as input_file:
        reader=csv.reader(input_file)

        # Loop on CSV rows, parsing each
        for i, row in enumerate(reader):

            # Skip the header row
            if i > 0:
                parsed_text['Source'].append('Gaggar')

                if row[1] == '0':
                    parsed_text['Synthetic'].append(0)
                    parsed_text['Author'].append('Human')
                    human_texts+=1

                if row[1] == '1':
                    parsed_text['Synthetic'].append(1)
                    parsed_text['Author'].append('GPT-3.5-turbo')
                    synthetic_texts+=1

                parsed_text['Text'].append(row[0])

    function_logger.info(f'Parsed Gaggar data: {human_texts + synthetic_texts} '+
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    ############
    # Yatsenko #
    ############

    # Load the dataset
    utils.disable_progress_bar()
    dataset=load_dataset(
        f'{config.RAW_DATA_PATH}/yatsenko/data',
        cache_dir=f'{config.RAW_DATA_PATH}/yatsenko'
    )

    # Counters
    human_texts=0
    synthetic_texts=0

    # Loop over and parse the dataset
    for i, record in enumerate(dataset['train']):

        parsed_text['Source'].append('Yatsenko')

        if record['source'] == 'human':
            parsed_text['Synthetic'].append(0)
            parsed_text['Author'].append('Human')
            human_texts+=1

        if record['source'] == 'ai':
            parsed_text['Synthetic'].append(1)
            parsed_text['Author'].append('Unknown model')
            synthetic_texts+=1

        parsed_text['Text'].append(record['text'])

    function_logger.info(
        'Parsed Yatsenko data: %s texts, %s human and %s synthetic',
        human_texts + synthetic_texts,
        human_texts,
        synthetic_texts
    )

    return parsed_text


def save_parsed_text(parsed_text: dict):
    '''Saves parsed and combined text data as single JSON
    file and parquet shards.'''

    # Get the logger
    function_logger=logging.getLogger(f'{__name__}.save_parsed_text')

    # Get some summary stats about the file
    total_texts=len(parsed_text['Synthetic'])
    synthetic_texts=sum(parsed_text['Synthetic'])
    human_texts=total_texts - synthetic_texts
    percent_synthetic=(synthetic_texts/total_texts)*100
    percent_human=(human_texts/total_texts)*100

    function_logger.info('Have %s texts', total_texts)
    function_logger.info('Human: %s(%s %%)', human_texts, percent_human)
    function_logger.info('Synthetic: %s(%s %%)',synthetic_texts, percent_synthetic)

    # Set up output directory
    output_directory=f'{config.INTERMEDIATE_DATA_PATH}'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    output_path=f'{output_directory}/all_texts.json'
    function_logger.info('Saving master to %s', output_path)

    # Save it as JSON
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(parsed_text, output_file, ensure_ascii=False, indent=4)

    # Convert to a dataframe
    data_df=pd.DataFrame(parsed_text)

    # Give it a shuffle
    data_df=data_df.sample(frac=1)

    # Do the test-train split
    training_df, testing_df=train_test_split(data_df, test_size=0.3)

    for data_df, dataset in zip([training_df, testing_df], ['train', 'test']):

        function_logger.info('Saving %s data', dataset)

        # Split the dataframe into two less than the number of cpus chunks
        chunks=np.array_split(data_df, mp.cpu_count() - 2)

        # Save each chunk as parquet with a clean index
        for i, chunk in enumerate(chunks):
            output_file=f'{config.INTERMEDIATE_DATA_PATH}/{dataset}_texts.{i}.parquet'
            chunk.reset_index(inplace=True, drop=True)
            chunk.to_parquet(output_file)
            function_logger.info('Saved %s', output_file)

    function_logger.info('Done')


def semantic_split() -> None:
    '''Main function to do semantic splitting of the parsed and sharded text.'''

    # Set-up multiprocess logging to file
    logfile=f'{config.LOG_PATH}/semantic_splitting.log'
    print(f'Will log to: {logfile}\n')

    logging_queue=mp.Manager().Queue(-1)

    log_listener=mp.Process(
        target=log_funcs.listener_process,
        args=(logging_queue, log_funcs.configure_listener, logfile)
    )

    log_listener.start()

    # Get logger for main process
    log_funcs.configure_worker(logging_queue)
    logger=logging.getLogger(f'{__name__}.semantic_split')
    logger.info('Main process started')

    # Set target lengths for splits in tokens
    target_lengths=[16,32,64,128,256,512]

    # Loop to process training and testing data separately
    for dataset in ['train', 'test']:

        logger.info('Running semantic splitting on %s data', dataset)

        # Collect the results
        chunks={
            'Text': [],
            'Synthetic': [],
            'Author': [],
            'Source': []
        }

        for target_length in target_lengths:
            logger.info('Splitting with target length %s tokens', target_length)

            # Get list of input files
            input_files=glob.glob(f'{config.INTERMEDIATE_DATA_PATH}/{dataset}_texts.*.parquet')

            # Instantiate pool with one worker per input file
            pool=mp.Pool(
                processes=len(input_files),
                maxtasksperchild=1
            )

            # Holder for returns from workers
            async_results=[]

            # Loop input files
            for i, data_file in enumerate(input_files):

                async_results.append(
                    pool.apply_async(
                        split_text,
                        args=(
                            data_file,
                            target_length,
                            i
                        )
                    )
                )

            # Clean up
            pool.close()
            pool.join()

            # Get the results
            results=[async_result.get() for async_result in async_results]

            # Collect the results
            for result in results:
                for key, value in result.items():
                    chunks[key].extend(value)

            logger.info('Finished target length %s', target_length)

        logger.info('Finished splitting %s data', dataset)

        # Convert to Pandas dataframe
        chunks_df=pd.DataFrame(chunks)

        # Give it a shuffle
        chunks_df=chunks_df.sample(frac=1)

        # Split the dataframe into shards
        chunk_shards=np.array_split(chunks_df, mp.cpu_count() - 2)

        # Save each chunk as parquet with a clean index
        for i, chunk in enumerate(chunk_shards):
            output_file=f'{config.INTERMEDIATE_DATA_PATH}/{dataset}_chunks.{i}.parquet'
            chunk.reset_index(inplace=True, drop=True)
            chunk.to_parquet(output_file)


def split_text(
        data_file: str=None,
        target_size: int=512,
        worker_num: int=0,
        sample_fraction: float=1
) -> dict:

    '''Function to parallelize semantic splitting of text over input files. 
    Meant to be called with multiprocessing worker. Take an input file 
    string, loads the data, splits sentences, collects results in dictionary
    and returns dictionary.'''

    # Set-up worker's logging
    logger=logging.getLogger(f'{__name__}.split_text')
    logger.info('Worker %s started', worker_num)

    data_df=pd.read_parquet(data_file)
    logger.info('Worker %s loaded %s', worker_num, os.path.basename(data_file))

    results={
        'Text': [],
        'Synthetic': [],
        'Author': [],
        'Source': []
    }

    # Tokenizer & splitter
    tokenizer_name='bert-base-uncased'
    tokenizer=Tokenizer.from_pretrained(tokenizer_name)
    splitter=TextSplitter.from_huggingface_tokenizer(tokenizer, target_size)

    # Calculate the number of records to process
    num_records=int(len(data_df)*sample_fraction)

    # Loop over records
    for i in range(num_records):

        text=data_df['Text'].iloc[i]
        chunks=splitter.chunks(text)

        for chunk in chunks:
            results['Text'].append(chunk)
            results['Synthetic'].append(data_df['Synthetic'].iloc[i])
            results['Author'].append(data_df['Author'].iloc[i])
            results['Source'].append(data_df['Source'].iloc[i])

    logger.info('Worker %s read %s records', worker_num, i+1)
    logger.info('Worker %s produced %s records', worker_num, len(results['Text']))

    return results
