'''Collection of functions for Luigi data pipeline & feature engineering'''

from __future__ import annotations

import numpy as np
import pandas as pd
# from typing import Tuple

# import json
# from statistics import mean, stdev
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

import classifier.configuration as config

def load_data() -> dict:
    '''Parses and combines perpplexity ratio scored text fraagments from
    all three Hans 2024 datasets.'''

    # Holder for each parsed datset
    dataframes = []

    for _, filename in config.SCORED_HANS_DATASETS.items():

        dataframe = pd.read_json(f'{config.HANS_DATA_PATH}/{filename}')

        # Replace and remove string 'OOM' and 'NAN' values
        dataframe.replace('NAN', np.nan, inplace = True)
        dataframe.replace('OOM', np.nan, inplace = True)
        dataframe.dropna(inplace = True)

        # Update name of score column, some earlier runs called
        # it by the old names
        dataframe.rename(columns = {
            'Binoculars score': 'Perplexity ratio score',
            'Observer peak memory (GB)': 'Reader peak memory (GB)',
            'Performer peak memory (GB)': 'Writer peak memory (GB)'
        }, inplace = True)

        # Fix some d-types
        dataframe = dataframe.astype({
            'Fragment': int,
            'Fragment length (tokens)': int,
            'Reader peak memory (GB)': float,
            'Writer peak memory (GB)': float,
            'Perplexity': float,
            'Cross-perplexity': float,
            'Perplexity ratio score': float
        })

        # get rid of some unnecessary columns
        dataframe.drop([
            'Fragment', 
            'Reader peak memory (GB)', 
            'Writer peak memory (GB)'
        ], axis = 1, inplace = True)

        dataframes.append(dataframe)

    # Combine the individual datasets and reset the index
    data_df = pd.concat(dataframes, axis = 0)
    data_df.reset_index(inplace = True, drop = True)

    # Split the data in to training and testing subsets.
    training_df = data_df.sample(frac = config.TRAIN_TEST_SPLIT, random_state = 42)
    testing_df = data_df.drop(training_df.index)
    training_df.reset_index(inplace = True, drop = True)
    testing_df.reset_index(inplace = True, drop = True)

    # Construct a single dictionary containing the JSON of the training
    # and testing dataframes
    results = {
        'training': training_df.to_json(),
        'testing': testing_df.to_json()
    }

    return results