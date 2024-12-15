# '''Classes to handle training and testing data for XGBoost classifier.'''

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# class TrainTestData:
#     '''Holds training and testing data'''

#     def __init__(self, training_data: str = None, testing_data: str = None) -> None:
        
#         # Add the training data
#         self.training = DataHolder(
#             data = pd.read_json(training_data)
#         )
        
#         # Add the testing data
#         self.testing = DataHolder(
#             data = pd.read_json(testing_data)
#         )

#         # Fit a standard scaler on the training data and
#         # add it to the DataHolders
#         standard_scaler = StandardScaler()
#         standard_scaler.fit(self.training.features)

#         self.training.standard_scaler = standard_scaler
#         self.testing.standard_scaler = standard_scaler


# class DataHolder:
#     '''Holds labels and features and related methods'''

#     def __init__(self, data: pd.DataFrame = None) -> None:

#         # Add the data
#         self.data = data
#         self.labels = self.data['Source'].to_numpy()
#         self.features = self.data.drop(['Source','Dataset', 'String'], axis = 1).to_numpy()

#         # Add and fit the label encoder/decoder
#         self.label_encoder = LabelEncoder()
#         self.label_encoder.fit(self.labels)

#         # Add the standard scaler later
#         self.standard_scaler = None

#     def encode_labels(self) -> np.ndarray:

#         return self.label_encoder.transform(self.labels)

#     def label_classes(self) -> None:

#         for i, class_name in enumerate(self.label_encoder.classes_):
#             print(f'{i}: {self.label_encoder.classes_[i]}')

#     def scale_features(self) -> np.ndarray:

#         return self.standard_scaler.transform(self.features)