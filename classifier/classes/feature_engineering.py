'''Classes related to feature engineering pipeline'''

import numpy as np
import pandas as pd
import configuration as config

class FeatureEngineering:
    '''Holds objects and methods for feature engineering pipeline.'''

    def __init__(self, raw_input_data_file_name: str = config.RAW_INPUT_DATA) -> None:

        # Set the string path/filename to the raw input data
        self.raw_input_data_file_name = raw_input_data_file_name

        # Load and clean the data
        self.load_and_clean_raw_data()

        # Train test split the data and parse by original dataset source
        self.parse_source()


    def load_and_clean_raw_data(self) -> None:
        '''Loads raw data into a Pandas dataframe, does
        basic clean-up and enforces data types'''

        # Load
        data_df = pd.read_json(self.raw_input_data_file_name)

        # Replace and remove string 'OOM' and 'NAN' values
        data_df.replace('NAN', np.nan, inplace = True)
        data_df.replace('OOM', np.nan, inplace = True)
        data_df.dropna(inplace = True)

        # Enforce dtypes
        data_df = data_df.astype({
            'Source record num': int,
            'Fragment length (words)': int,
            'Fragment length (tokens)': int,
            'Dataset': str,
            'Source': str,
            'String': str,
            'Perplexity': float,
            'Cross-perplexity': float,
            'Perplexity ratio score': float,
            'Reader time (seconds)': float,
            'Writer time (seconds)': float,
            'Reader peak memory (GB)': float,
            'Writer peak memory (GB)': float
        })

        # Shuffle the deck, resetting the index
        data_df = data_df.sample(frac = 1).reset_index(drop = True)
        data_df.reset_index(inplace = True, drop = True)

        self.all = AllDataHolder(data_df)

    def parse_source(self) -> None:
        '''Splits data by original source dataset.

        Going to build a few levels of dictionary structure here
        and set it up so we can access it with dot notation. Here is
        the structure:
        
        data (FeatureEngineering class instance)
        |
        |--all
        |
        |--dataset_names
        |--training
        |  |
        |  |--all
        |  |  |
        |  |  |--human
        |  |  |--falcon
        |  |  |--llama
        |  |
        |  |--cc_news
        |  |  |
        |  |  |--human
        |  |  |--falcon
        |  |  |--llama
        |  |
        |  |--cnn
        |  |  |
        |  |  |--human
        |  |  |--falcon
        |  |  |--llama
        |  |
        |  |--pub_med
        |     |
        |     |--human
        |     |--falcon
        |     |--llama
        |
        |--testing
        |  |
        ...
        '''

        # Get unique values from the dataset column
        self.dataset_names = self.all.combined['Dataset'].unique()

        # Make train/test split and place each into instance of TrainTestSplitHolder class
        training = self.all.combined.sample(frac = 0.8, random_state = 42)
        testing = self.all.combined.drop(training.index)

        training.reset_index(inplace = True, drop = True)
        testing.reset_index(inplace = True, drop = True)

        self.training = TrainTestSplitHolder(training, self.dataset_names)
        self.testing = TrainTestSplitHolder(testing, self.dataset_names)

class TrainTestSplitHolder:
    '''Holds training or testing data from train test split.'''

    def __init__(self, data: pd.DataFrame = None, dataset_names: list = None) -> None:

        # Add the complete dataset
        self.combined = data

        # Add subsets for data from each original dataset source
        for dataset_name in dataset_names:

            # Get data subset
            dataset_data = data[data['Dataset'] == dataset_name]
            setattr(self, dataset_name, dataset_data)


class AllDataHolder:
    '''Holds master copy of data before train test split.
    has three attributes: combined, human, synthetic.'''

    def __init__(self, data: pd.DataFrame = None) -> None:

        self.combined = data
        self.human = data[data['Source'] == 'human']
        self.synthetic = data[data['Source'] == 'synthetic']