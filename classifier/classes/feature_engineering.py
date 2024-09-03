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

        self.all = DataHolder(data_df)

    def parse_source(self) -> None:
        '''Splits data by original source dataset.

        Going to build a few levels of dictionary structure here
        and set it up so we can access it with dot notation. Here is
        the structure:
        
        data (FeatureEngineering class instance)
        |
        |--raw_input_data_file_name
        |--dataset_names
        |--generation_models
        |
        |--all
        |  |
        |  |--combined
        |  |--human
        |  |--synthetic_combined
        |  |--falcon7
        |  |--llama2
        |
        |--training
        |  |
        |  |--all
        |  |  |
        |  |  |--combined
        |  |  |--human
        |  |  |--synthetic_combined
        |  |  |--falcon7
        |  |  |--llama2
        |  |
        |  |--cc_news
        |  |  |
        |  |  |--combined
        |  |  |--human
        |  |  |--synthetic_combined
        |  |  |--falcon7
        |  |  |--llama2
        |  |
        |  |--cnn
        |  |  |
        |  |  |--combined
        |  |  |--human
        |  |  |--synthetic_combined
        |  |  |--falcon7
        |  |  |--llama2
        |  |
        |  |--pub_med
        |     |
        |     |--combined
        |     |--human
        |     |--synthetic_combined
        |     |--falcon7
        |     |--llama2
        |
        |--testing
        |  |
        |  |--...
        '''

        # Get unique values from the dataset column
        self.dataset_names = list(self.all.combined['Dataset'].unique())

        # Get unique values from the generator column. This will be either
        # 'human' or the name of the model that generated the fragment
        # if it is synthetic
        self.generation_models = list(self.all.combined['Generator'].unique())

        # Remove 'human' so we get just the models
        self.generation_models.remove('human')

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
        self.all = DataHolder(data)

        # Add subsets for data from each original dataset source
        for dataset_name in dataset_names:

            # Get data subset
            dataset_data = data[data['Dataset'] == dataset_name]

            # Add it to the class by name
            setattr(self, dataset_name, DataHolder(dataset_data))


class DataHolder:
    '''Holds master copy of data in attributes: 
    combined, synthetic_combined, human, and one for each model.'''

    def __init__(self, data: pd.DataFrame = None) -> None:

        self.combined = data
        self.synthetic_combined = data[data['Source'] == 'synthetic']

        # Get all of the generation sources present in the data, including
        # human so we can add the data for each on individually.
        generation_sources = data['Generator'].unique()

        for generation_source in generation_sources:

            # Get data subset
            generation_source_data = data[data['Generator'] == generation_source]

            # Add it to the class by name
            setattr(self, generation_source, generation_source_data)