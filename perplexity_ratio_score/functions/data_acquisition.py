'''Collection of functions to download, parse, combine and semantically split
text data sets for perplexity ratio scoring.'''

# Standard library imports
#import glob
import csv
import json
#import os.path
import zipfile
import urllib.request
from pathlib import Path
from itertools import product

# PyPI imports
#import pyarrow
import kaggle
import numpy as np
import pandas as pd
from datasets import load_dataset, utils

# Internal imports
import perplexity_ratio_score.configuration as config

def get_data():
    '''Main function to run steps in data acquisition pipeline.'''

    _=download_raw_data()
    _=parse_raw_data()


def download_raw_data():
    '''Downloads raw data from internet sources'''

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

    print('Finished downloading Hans data')

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

    print('Finished downloading Gerami data')

    ############
    # Grinberg #
    ############

    # Set up output directory
    output_directory=f'{config.RAW_DATA_PATH}/grinberg'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Output file
    output_file=f'{output_directory}/human-vs-llm-text-corpus.zip'

    # Only download the file if we don't already have it
    #if Path(output_file).is_file() is False:
    kaggle.api.dataset_download_files(
        'starblasters8/human-vs-llm-text-corpus',
        path=output_directory
    )

    # Unzip the data
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    print('Finished downloading Grinberg data')

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

    print('Finished downloading Gaggar data')

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

    print('Finished downloading Yatsenko data')

    return True


def parse_raw_data():
    '''Parses and combines text from each data set'''

    # Holder for results
    parsed_text={
        'text': [],
        'synthetic': [],
        'author': [],
        'source': []
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
                parsed_text['source'].append('hans')
                parsed_text['synthetic'].append(1)
                parsed_text['author'].append(generating_model)

                if generating_model == 'llama2_13':
                    text=data['meta-llama-Llama-2-13b-hf_generated_text_wo_prompt']

                elif generating_model == 'falcon7':
                    text=data['-fs-cml-models-Falcon-falcon-7b_generated_text_wo_prompt']

                parsed_text['text'].append(text)

                synthetic_texts+=1

                # Get the human text and add to parsed text
                parsed_text['source'].append('hans')
                parsed_text['synthetic'].append(0)
                parsed_text['author'].append('human')

                if 'article' in data.keys():
                    text=data['article']

                elif 'text' in data.keys():
                    text=data['text']

                parsed_text['text'].append(text)

                human_texts+=1

    print(f'\nParsed Hans data: {human_texts + synthetic_texts} '+
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
                parsed_text['source'].append('gerami')

                if row[1] == '0.0':
                    parsed_text['synthetic'].append(0)
                    parsed_text['author'].append('human')
                    human_texts+=1

                if row[1] == '1.0':
                    parsed_text['synthetic'].append(1)
                    parsed_text['author'].append('unknown_model')
                    synthetic_texts+=1

                parsed_text['text'].append(row[0])

    print(f'Parsed Gerami data: {human_texts + synthetic_texts} '+
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
        parsed_text['source'].append('grinberg')

        if source == 'Human':
            parsed_text['synthetic'].append(0)
            parsed_text['author'].append('human')
            human_texts+=1

        if source != 'Human':
            parsed_text['synthetic'].append(1)
            parsed_text['author'].append('unknown_model')
            synthetic_texts+=1

        parsed_text['text'].append(text)

    print(f'Parsed Grinberg data: {human_texts + synthetic_texts} '+
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
                parsed_text['source'].append('gaggar')

                if row[1] == '0':
                    parsed_text['synthetic'].append(0)
                    parsed_text['author'].append('human')
                    human_texts+=1

                if row[1] == '1':
                    parsed_text['synthetic'].append(1)
                    parsed_text['author'].append('GPT-3.5-turbo')
                    synthetic_texts+=1

                parsed_text['text'].append(row[0])

    print(f'Parsed Gaggar data: {human_texts + synthetic_texts} '+
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    ############
    # Yatsenko #
    ############

    # Load the dataset
    utils.disable_progress_bar()
    dataset=load_dataset(f'{config.RAW_DATA_PATH}/yatsenko/data')

    # Counters
    human_texts=0
    synthetic_texts=0

    # Loop over and parse the dataset
    for i, record in enumerate(dataset['train']):

        parsed_text['source'].append('yatsenko')

        if record['source'] == 'human':
            parsed_text['synthetic'].append(0)
            parsed_text['author'].append('human')
            human_texts+=1

        if record['source'] == 'ai':
            parsed_text['synthetic'].append(1)
            parsed_text['author'].append('unknown_model')
            synthetic_texts+=1

        parsed_text['text'].append(record['text'])

    print(f'Parsed Yatsenko data: {human_texts + synthetic_texts} '
        f'texts, {human_texts} human and {synthetic_texts} synthetic')

    return parsed_text


def save_parsed_data(parsed_text: dict):
    '''Saves parsed and combined text data as single JSON
    file and parquet shards.'''

    # Get some summary stats about the file
    total_texts=len(parsed_text['synthetic'])
    synthetic_texts=sum(parsed_text['synthetic'])
    human_texts=total_texts - synthetic_texts
    percent_synthetic=(synthetic_texts/total_texts)*100
    percent_human=(human_texts/total_texts)*100

    print(f'\nHave {total_texts} texts')
    print(f' Human: {human_texts}({percent_human:.1f}%)')
    print(f' Synthetic: {synthetic_texts}({percent_synthetic:.1f}%)')

    # Set up output directory
    output_directory=f'{config.INTERMEDIATE_DATA_PATH}'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Save it as JSON
    with open(f'{output_directory}/all_texts.json', 'w', encoding='utf-8') as output_file:
        json.dump(parsed_text, output_file, ensure_ascii=False, indent=4)

    # Convert to a dataframe
    data_df=pd.DataFrame(parsed_text)
    # Give it a shuffle
    data_df=data_df.sample(frac=1)

    # Split the dataframe into 16 chunks
    chunks=np.array_split(data_df, 16)

    # Save each chunk as parquet with a clean index
    for i, chunk in enumerate(chunks):
        output_file=f'{config.INTERMEDIATE_DATA_PATH}/texts.{i}.parquet'
        chunk.reset_index(inplace=True, drop=True)
        chunk.to_parquet(output_file)
