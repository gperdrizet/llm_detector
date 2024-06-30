'''Collection of functions for data manipulation, parsing and exploration'''

from __future__ import annotations
from typing import Tuple

import json
from statistics import mean, stdev
import pandas as pd

def parse_hans_data(
    hans_datasets: dict,
    hans_data: dict,
    hans_metadata,
    binoculars_data_path
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Parses and collects datasets from Hans et al (2024) Binocular publication. 
    Takes dict describing datasets with names and file paths and two empty dicts 
    to hold output: one for metadata and one for the collected datasets. Returns 
    parses and returns populated data and metadata pandas dataframes.'''

    # The hans datasets use different keys for the human text, use this
    # dict. to look up the correct one based on the data source
    human_text_keys={
        'cc_news': 'text',
        'cnn': 'article',
        'pubmed': 'article'
    }

    # Loop on outer level of hand datasets dict.
    for generation_model, datasets in hans_datasets.items():

        # loop on each inner level of hans datasets dict
        for data_source, datafile_name in datasets.items():

            # Build the absolute file name for this dataset
            dataset_file=f'{binoculars_data_path}/{datafile_name}'

            # Set initial values for some collector vars
            record_count=1
            human_text_lengths=[]
            synthetic_text_lengths=[]
            human_text_fractions=[]

            # Open the data file and loop on lines, loading each as JSON
            with open(dataset_file, encoding='utf-8') as f:
                for line in f:
                    record=json.loads(line)

                    # If we can find a text record in the JSON object, continue processing
                    if human_text_keys[data_source] in list(record.keys()):

                        # Get the human and synthetic text
                        human_text=record[human_text_keys[data_source]]
                        synthetic_text=record[list(record.keys())[-1]]

                        # Get the texts' lengths
                        human_text_length=len(human_text.split(' '))
                        synthetic_text_length=len(synthetic_text.split(' '))

                        # Collect the texts' lengths
                        human_text_lengths.append(human_text_length)
                        synthetic_text_lengths.append(synthetic_text_length)

                        # Get and collect the fraction of this record's text that is human
                        total_text_length=human_text_length + synthetic_text_length
                        human_text_fraction=human_text_length / total_text_length
                        human_text_fractions.append(human_text_fraction)

                        # Count this record
                        record_count+=1

                        # Add data from this record to the collected data result
                        hans_data['Record ID'].append(record_count)
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

    hans_data_df=pd.DataFrame.from_dict(hans_data)
    hans_metadata_df=pd.DataFrame.from_dict(hans_metadata)

    return hans_metadata_df, hans_data_df
