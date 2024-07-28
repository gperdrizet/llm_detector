'''Classes to handle training and testing data for XGBoost classifier.'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TrainTestData:
    '''Holds training and testing data'''

    def __init__(self, training_data: str = None, testing_data: str = None) -> None:
        
        # Add the training data
        self.training = DataHolder(
            data = pd.read_json(training_data)
        )
        
        # Add the testing data
        self.testing = DataHolder(
            data = pd.read_json(testing_data)
        )

class DataHolder:
    '''Holds labels and features and related methods'''

    def __init__(self, data: pd.DataFrame = None) -> None:

        # Add the data
        self.data = data
        self.labels = self.data['Source'].to_numpy()
        self.features = self.data.drop(['Source','Dataset', 'String'], axis = 1).to_numpy()

        # Add the label encoder/decoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    def encode_labels(self) -> np.ndarray:

        return self.label_encoder.transform(self.labels)