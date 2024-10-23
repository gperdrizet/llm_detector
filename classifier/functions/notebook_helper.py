'''Collection of functions refactored from notebooks for data handling'''

from __future__ import annotations
from typing import Callable

import pickle
import re
import nltk
import pathlib
import logging
import time
import multiprocessing as mp
import cupy as cp
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import transformers

from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.neighbors import KernelDensity

from math import log2
from statistics import mean
from scipy.stats import ttest_ind, exponnorm, fit, gaussian_kde
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import configuration as config
import classes.llm as llm_class


# Set cupy CUDA device to GPU 0 (this is the GTX1070 on pyrite)
#cp.cuda.Device(0).use()


def start_logger(
        logfile_name: str='llm_detector.log',
        logger_name: str='benchmarking'
) -> Callable:

    '''Sets up logging, returns logger'''

    # Build logfile name
    logfile = f'{config.LOG_PATH}/{logfile_name}'
    print(f'Will log to: {logfile}\n')

    # Clear logs from previous runs
    pathlib.Path(logfile).unlink(missing_ok=True)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(config.LOG_LEVEL)

    handler = logging.handlers.RotatingFileHandler(
        f'{config.LOG_PATH}/{logfile_name}',
        encoding = 'utf-8',
        maxBytes = 1 * 1024 * 1024,  # 1 MiB
        backupCount = 5
    )

    formatter = logging.Formatter(config.LOG_PREFIX,
                                  datefmt = '%Y-%m-%d %I:%M:%S %p')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def mean_difference_ci(data_df: pd.DataFrame) -> None:
    '''Conducts t-test on difference in perplexity ratio score means on
    length binned data. Prints, means, difference in means, 95% confidence 
    interval around the difference in means and the p-value for the
    difference in means.
    '''

    # Bin the data
    binned_data = data_df[['Fragment length (tokens)', 'Perplexity ratio score', 'Source']].copy()
    bins = pd.cut(binned_data.loc[:, 'Fragment length (tokens)'], 10)
    binned_data.loc[:, 'Length bin (tokens)'] = bins

    # Make the length bin column string and get the unique values
    binned_data = binned_data.astype({'Length bin (tokens)': str})
    length_bins = binned_data.loc[:, 'Length bin (tokens)'].unique()

    # Loop on the bins run t-test
    for length_bin in length_bins:

        # Get the human and synthetic data for this length bin
        human_prs = binned_data.loc[:, 'Perplexity ratio score'][(binned_data.loc[:, 'Length bin (tokens)'] == length_bin) & (binned_data.loc[:, 'Source'] == 'human') ]
        synthetic_prs = binned_data.loc[:, 'Perplexity ratio score'][(binned_data.loc[:, 'Length bin (tokens)'] == length_bin) & (binned_data.loc[:, 'Source'] == 'synthetic') ]

        # Make sure there is human and synthetic data to work wit in this bin
        if len(human_prs) != 0 and len(synthetic_prs) != 0:

            # Get the means in question and their difference
            human_mean = mean(human_prs)
            synthetic_mean = mean(synthetic_prs)
            mean_diff = human_mean - synthetic_mean

            cm = sms.CompareMeans(sms.DescrStatsW(human_prs), sms.DescrStatsW(synthetic_prs))
            difference = cm.tconfint_diff(usevar='unequal')
            low_bound = difference[0]
            high_bound = difference[1]

            ttest_result = ttest_ind(human_prs, synthetic_prs, alternative='greater')

            print(f'Length bin: {length_bin} tokens')
            print(f'  Human mean: {human_mean:.3f}, synthetic mean: {synthetic_mean:.3f}')
            print(f'  Difference in means = {mean_diff:.3f}, 95% CI = ({low_bound:.3f}, {high_bound:.3f})')
            print(f'  p-value (human > synthetic) = {ttest_result.pvalue}\n')


def make_labels(training_df, testing_df):
    '''Takes training and testing dataframes, gets and encode human/synthetic
    labels and returns.'''

    # Get the labels
    training_labels = training_df['Source']
    testing_labels = testing_df['Source']

    # Encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(training_labels)
    training_labels = pd.Series(label_encoder.transform(training_labels)).astype(np.int64)
    testing_labels = pd.Series(label_encoder.transform(testing_labels)).astype(np.int64)

    return training_labels, testing_labels


def exp_gaussian_fit(
        scores: np.ndarray = None, 
        bounds: list[list[float, float]] = [[0.001, 1.0], [0.001, 1.0], [0.001, 1.0]]
) -> Callable:
    
    '''Fit and return exponnorm on scores'''

    # Do the fit
    exponnorm_fit = fit(exponnorm, scores.astype(np.float64), bounds = bounds)

    # Build function from fit
    exponnorm_func = exponnorm(exponnorm_fit.params.K, exponnorm_fit.params.loc, exponnorm_fit.params.scale)

    # Print the fitted parameters
    print(f'  Shape: {exponnorm_fit.params.K}')
    print(f'  Mean: {exponnorm_fit.params.loc}')
    print(f'  Variance: {exponnorm_fit.params.scale}')

    # Return the function
    return exponnorm_func


def kl_divergence(p: list, q: list) -> np.ndarray:
    '''Takes two lists, calculates Kullback-Leibler divergence.'''

    # Convert inputs to numpy
    p = np.asarray(p)
    q = np.asarray(q)

    # Set handling for overflows/underflows - just ignore. We will handle infinite
    # or nan values later by just filtering them out.
    with np.errstate(over = 'ignore', under = 'ignore', divide = 'ignore', invalid = 'ignore'):
        kld_values = p * np.log2(p/q)

    return kld_values


def get_kl_kde(
        figure_title: str = None,
        scores: pd.Series = None,
        human_fit_func: Callable = None,
        synthetic_fit_func: Callable = None,
        padding: float = None,
        sample_frequency: float = None
) -> tuple[gaussian_kde, gaussian_kde, Callable]:
    
    '''Get kernel density estimate of Kullback-Leibler divergence'''

    # Get a list of points covering the range of score values and extend
    # the left and right edges a little bit, otherwise the kernel density
    # estimate tends to droop at the edges for the range. We will clip
    # the padding off later.
    x = np.arange(min(scores) - padding, max(scores) + padding, sample_frequency).tolist()
    print(f'Will calculate {len(x)} fitted values')

    # Get fitted values for the points
    human_fitted_values = human_fit_func.pdf(x)
    synthetic_fitted_values = synthetic_fit_func.pdf(x)

    # Calculate the KL divergences of the fitted values
    synthetic_human_kld = kl_divergence(synthetic_fitted_values, human_fitted_values)
    human_synthetic_kld = kl_divergence(human_fitted_values, synthetic_fitted_values)

    # Get rid of any np.nan, without changing the length
    mask = np.isnan(synthetic_human_kld)
    synthetic_human_kld[mask] = 0

    mask = np.isnan(human_synthetic_kld)
    human_synthetic_kld[mask] = 0

    # Get rid of any inf without changing the length
    mask = np.isinf(synthetic_human_kld)
    synthetic_human_kld[mask] = 0

    mask = np.isinf(human_synthetic_kld)
    human_synthetic_kld[mask] = 0

    # Convert the kl 'density' values into integer 'count' values
    synthetic_human_kld = synthetic_human_kld + abs(min(synthetic_human_kld))
    synthetic_human_kld = synthetic_human_kld * 100
    synthetic_human_kld_counts = [int(density) for density in synthetic_human_kld]

    print(f'Min synthetic-human KLD count value {min(synthetic_human_kld_counts)}')
    print(f'Max synthetic-human KLD count value: {max(synthetic_human_kld_counts)}')

    human_synthetic_kld = human_synthetic_kld + abs(min(human_synthetic_kld))
    human_synthetic_kld = human_synthetic_kld * 100
    human_synthetic_kld_counts = [int(density) for density in human_synthetic_kld]

    print(f'Min human-synthetic KLD count value {min(human_synthetic_kld_counts)}')
    print(f'Max human-synthetic KLD count value: {max(human_synthetic_kld_counts)}')

    # Now, construct a list where each value of x appears a number of times
    # equal to it's kld 'count'
    synthetic_human_kld_scores = []
    human_synthetic_kld_scores = []

    for i in range(len(synthetic_human_kld_counts)):
        synthetic_human_kld_scores.extend([x[i]] * synthetic_human_kld_counts[i])

    for i in range(len(human_synthetic_kld_counts)):
        human_synthetic_kld_scores.extend([x[i]] * human_synthetic_kld_counts[i])

    # Then, run a KDE on the reconstructed KL scores
    synthetic_human_kld_kde = gaussian_kde(synthetic_human_kld_scores)
    human_synthetic_kld_kde = gaussian_kde(human_synthetic_kld_scores)

    # Finally, use the PDF to get density for x after re-clipping x to the
    # range of the original data
    clipped_x = []
    clipped_synthetic_human_kld = []
    clipped_synthetic_human_kld_counts = []
    clipped_human_synthetic_kld = []
    clipped_human_synthetic_kld_counts = []

    for i, j in enumerate(x):
        if j > min(scores) and j < max(scores):

            clipped_x.append(j)

            clipped_synthetic_human_kld.append(synthetic_human_kld[i])
            clipped_human_synthetic_kld.append(human_synthetic_kld[i])

            clipped_synthetic_human_kld_counts.append(synthetic_human_kld_counts[i])
            clipped_human_synthetic_kld_counts.append(human_synthetic_kld_counts[i])

    clipped_synthetic_human_kld_kde_values = synthetic_human_kld_kde.pdf(clipped_x)
    clipped_human_synthetic_kld_kde_values = human_synthetic_kld_kde.pdf(clipped_x)

    fig, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.3}
    )

    fig.suptitle(figure_title, fontsize='x-large')

    axs[0,0].set_title('KL divergence density')
    axs[0,0].scatter(clipped_x, clipped_synthetic_human_kld, s = 1, label = 'synthetic-human')
    axs[0,0].scatter(clipped_x, clipped_human_synthetic_kld, s = 1, label = 'human-synthetic')
    axs[0,0].set_xlabel('Score')
    axs[0,0].set_ylabel('Density')
    axs[0,0].legend(loc = 'upper right', fontsize = 'small', markerscale = 5)

    axs[0,1].set_title('KL divergence counts')
    axs[0,1].scatter(clipped_x, clipped_synthetic_human_kld_counts, s = 1, label = 'synthetic-human')
    axs[0,1].scatter(clipped_x, clipped_human_synthetic_kld_counts, s = 1, label = 'human-synthetic')
    axs[0,1].set_xlabel('Score')
    axs[0,1].set_ylabel('Count')
    axs[0,1].legend(loc = 'upper right', fontsize = 'small', markerscale = 5)

    axs[1,0].set_title('KL score counts')
    axs[1,0].hist(synthetic_human_kld_scores, bins = 50, density = True, alpha = 0.5, label = 'synthetic-human')
    axs[1,0].hist(human_synthetic_kld_scores, bins = 50, density = True, alpha = 0.5, label = 'human-synthetic')
    axs[1,0].set_xlabel('Score')
    axs[1,0].set_ylabel('Count')
    axs[1,0].legend(loc = 'upper right', fontsize = 'small')

    axs[1,1].set_title('KL KDE')
    axs[1,1].scatter(clipped_x, clipped_synthetic_human_kld_kde_values, s = 1, label = 'synthetic-human')
    axs[1,1].scatter(clipped_x, clipped_human_synthetic_kld_kde_values, s = 1, label = 'human-synthetic')
    axs[1,1].set_xlabel('Score')
    axs[1,1].set_ylabel('Count')
    axs[1,1].legend(loc = 'upper right', fontsize = 'small', markerscale = 5)

    return (synthetic_human_kld_kde, human_synthetic_kld_kde, plt)


nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 


def clean_text(text: str = None) -> str:
    '''Cleans up text string for TF-IDF analysis.'''
    
    # Lowercase everything
    text = text.lower()

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # Remove URLs 
    text = re.sub(r"http\S+", "",text)
    
    # Remove html tags
    html = re.compile(r'<.*?>') 
    text = html.sub(r'',text)
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

    # Remove punctuations
    for p in punctuations:
        text = text.replace(p,'')
        
    # Remove stopwords
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    return text


def score_known_text_fragments(data_df: pd.DataFrame, tfidf_luts: dict = None) -> dict:
    '''Scores text fragments with product normalized difference in
    log2 TF-IDF mean.'''

    # Holders for TF-IDF values
    product_normalized_human_dmean_tfidf = []
    product_normalized_synthetic_dmean_tfidf = []

    # Loop on dataframe rows
    for _, row in data_df.iterrows():
        
        human_tfidf_sum = 0
        synthetic_tfidf_sum = 0

        # Get the text from this row
        text = row['String']

        # Clean the text
        text = clean_text(text)

        # Split the text into words
        words = text.split(' ')

        # Score the words using the human and synthetic luts
        for word in words:

            if word in tfidf_luts['human'].keys():
                human_tfidf_sum += tfidf_luts['human'][word]

            if word in tfidf_luts['synthetic'].keys():
                synthetic_tfidf_sum += tfidf_luts['synthetic'][word]

        # Get the means
        human_tfidf_mean = human_tfidf_sum / len(words)
        synthetic_tfidf_mean = synthetic_tfidf_sum / len(words)
        dmean_tfidf = human_tfidf_mean - synthetic_tfidf_mean
        product_normalized_dmean_tfidf = dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

        if row['Source'] == 'human':
            product_normalized_human_dmean_tfidf.append(product_normalized_dmean_tfidf)

        elif row['Source'] == 'synthetic':
            product_normalized_synthetic_dmean_tfidf.append(product_normalized_dmean_tfidf)

    return {'human': product_normalized_human_dmean_tfidf, 'synthetic': product_normalized_synthetic_dmean_tfidf}


def score_text_fragments(data_df: pd.DataFrame, tfidf_luts: dict = None) -> dict:
    '''Scores text fragments, returns human and synthetic TF-IDF and product 
    normalized difference in log2 TF-IDF mean'''

    # Holders for new features
    tfidf_scores = []
    human_tfidf = []
    synthetic_tfidf = []

    # Loop on dataframe rows
    for _, row in data_df.iterrows():
        
        human_tfidf_sum = 0
        synthetic_tfidf_sum = 0

        # Get the text from this row
        text = row['String']

        # Clean the text
        text = clean_text(text)

        # Split the text into words
        words = text.split(' ')

        # Score the words using the human and synthetic luts
        for word in words:

            if word in tfidf_luts['human'].keys():
                human_tfidf_sum += tfidf_luts['human'][word]

            if word in tfidf_luts['synthetic'].keys():
                synthetic_tfidf_sum += tfidf_luts['synthetic'][word]

        # Get the means
        human_tfidf_mean = human_tfidf_sum / len(words)
        synthetic_tfidf_mean = synthetic_tfidf_sum / len(words)
        dmean_tfidf = human_tfidf_mean - synthetic_tfidf_mean
        product_normalized_dmean_tfidf = dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

        human_tfidf.append(human_tfidf_mean)
        synthetic_tfidf.append(synthetic_tfidf_mean)
        tfidf_scores.append(product_normalized_dmean_tfidf)

    return {'human_tfidf': human_tfidf, 'synthetic_tfidf': synthetic_tfidf, 'tfidf_score': tfidf_scores}


# # def prep_training_data(data):
# #     '''Takes instance of feature engineering class, prepares data for classifier training'''

# #     # Retrieve training and testing data
# #     training_data_df = data.training.all.combined.copy()
# #     testing_data_df = data.testing.all.combined.copy()

# #     # Remove rows containing NAN
# #     training_data_df.dropna(inplace = True)
# #     testing_data_df.dropna(inplace = True)

# #     # Drop unnecessary or un-trainable features
# #     feature_drops = [
# #         'Source record num',
# #         'Dataset',
# #         'Generator',
# #         'String',
# #         'Reader time (seconds)',
# #         'Writer time (seconds)',
# #         'Reader peak memory (GB)',
# #         'Writer peak memory (GB)'
# #     ]

# #     training_data_df.drop(feature_drops, axis = 1, inplace = True)
# #     testing_data_df.drop(feature_drops, axis = 1, inplace = True)

# #     # Split the data into features and labels
# #     labels_train = training_data_df['Source']
# #     features_train_df = training_data_df.drop('Source', axis = 1)

# #     labels_test = testing_data_df['Source']
# #     features_test_df = testing_data_df.drop('Source', axis = 1)

# #     # Encode string class values as integers
# #     label_encoder = LabelEncoder()
# #     label_encoder = label_encoder.fit(labels_train)
# #     labels_train = label_encoder.transform(labels_train)
# #     labels_test = label_encoder.transform(labels_test)

# #     print(f'Training data: {len(labels_train)} examples')
# #     print(f'Test data: {len(labels_test)} examples')

# #     return features_train_df, features_test_df, labels_train, labels_test


def add_poly_features(
        features_train_df: pd.DataFrame,
        features_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Adds second order polynomial features, returns numpy array'''

    # Instantiate polynomial features instance
    trans = PolynomialFeatures(degree = 2)

    # Fit on and transform training data
    poly_features_train = trans.fit_transform(features_train_df)

    # Use the scaler fit from the training data to transform the test data
    poly_features_test = trans.fit_transform(features_test_df)

    print(f'Polynomial training data shape: {poly_features_train.shape}')
    print(f'Polynomial testing data shape: {poly_features_test.shape}')

    return poly_features_train, poly_features_test


def add_spline_features(
        features_train_df: pd.DataFrame,
        features_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Adds third order spline features, returns numpy array'''

    # Instantiate polynomial features instance
    trans = SplineTransformer()

    # Fit on and transform training data
    spline_features_train = trans.fit_transform(features_train_df)

    # Use the scaler fit from the training data to transform the test data
    spline_features_test = trans.fit_transform(features_test_df)

    print(f'Spline training data shape: {spline_features_train.shape}')
    print(f'Spline testing data shape: {spline_features_test.shape}')

    return spline_features_train, spline_features_test


def add_cv_scores(results: dict, scores: dict, condition: str) -> dict:
    '''Adds results of sklearn cross-validation run to
    results data structure, returns updated results.'''

    # Figure out how many folds we are adding
    num_folds = len(scores['fit_time'])
    
    # Add the fold numbers
    results['Fold'].extend(list(range(num_folds)))

    # Add the condition description
    results['Condition'].extend([condition] * num_folds)

    # Add the fit times
    results['Fit time (sec.)'].extend(scores['fit_time'])

    # Add the scores
    results['Accuracy (%)'].extend(scores['test_accuracy'])
    results['False positive rate'].extend(scores['test_false_positive_rate'])
    results['False negative rate'].extend(scores['test_false_negative_rate'])
    results['Binary cross-entropy'].extend(scores['test_binary_cross_entropy'])

    return results


def add_two_factor_cv_scores(
        results: dict, 
        scores: dict, 
        condition: str, 
        optimization: str
) -> dict:
    
    '''Adds results of sklearn cross-validation run to
    results data structure, returns updated results.'''

    # Figure out how many folds we are adding
    num_folds = len(scores['test_accuracy'])
    
    # Add the fold numbers
    results['Fold'].extend(list(range(num_folds)))

    # Add the condition description
    results['Condition'].extend([condition] * num_folds)

    # Add the optimization
    results['Optimized'].extend([optimization] * num_folds)

    # Add the scores
    results['Accuracy (%)'].extend(scores['test_accuracy'])
    results['False positive rate'].extend(scores['test_false_positive_rate'])
    results['False negative rate'].extend(scores['test_false_negative_rate'])
    results['Binary cross-entropy'].extend(scores['test_binary_cross_entropy'])

    return results


def percent_accuracy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns mean accuracy
    of predictions as a percent.'''

    # Get the scikit-learn normalized accuracy score
    accuracy = accuracy_score(labels, predictions, normalize = True) * 100

    return accuracy


def binary_cross_entropy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns non-negated
    log loss for binary classification.'''

    # Get the scikit-learn normalized accuracy score
    log_loss_score = log_loss(labels, predictions, normalize = True)

    return log_loss_score


def negated_binary_cross_entropy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns negated log loss 
    for binary classification. For use in situations where 
    'larger-is-better is desirable'''

    # Get the scikit-learn normalized accuracy score
    log_loss_score = -1 * log_loss(labels, predictions, normalize = True)

    return log_loss_score


def false_positive_rate(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. returns false positive
    rate from scikit-learn confusion matrix.'''

    # Extract counts from confusion matrix
    _, fp, _, tp = confusion_matrix(labels, predictions).ravel()

    # Calculate the false positive rate, protecting from division by zero
    if tp + fp == 0:
        false_positive_rate = 0

    else:
        false_positive_rate = fp / (tp + fp)

    return false_positive_rate


def false_negative_rate(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Scoring function for use with scikit-learn make_scorer take
    return false_positive_rates a model, features and labels. Returns 
    false negative rate from scikit-learn confusion matrix.'''

    # Extract counts from confusion matrix
    tn, _, fn, _ = confusion_matrix(labels, predictions).ravel()

    # Calculate the false negative rate, protecting from division by zero
    if tn + fn == 0:
        false_negative_rate = 0

    else:
        false_negative_rate = fn / (tn + fn)

    return false_negative_rate


def hyperopt(
      params: dict = None,
      model = None,
      features_training: np.ndarray = None,
      labels_training: np.ndarray = None,
      features_validation: np.ndarray = None,
      labels_validation: np.ndarray = None
) -> float:
   
    '''Cross validate a classifier with a set of hyperparameters, using a 
    single, static training/validation split and early stopping. Return scores.'''

    # Let XGBoost use the GPU
    if isinstance(model, XGBClassifier):

        # Set model parameters
        model.set_params(device = 'cuda', **params)

        # Fit the model
        model.fit(cp.array(features_training), cp.array(labels_training))

        # Make predictions for validation data
        labels_predicted = model.predict(cp.array(features_validation))

        # Return the binary cross-entropy
        return log_loss(labels_validation, labels_predicted)

    else:

        # Set model parameters
        model.set_params(**params)

        # Fit the model
        model.fit(features_training, labels_training)

        # Make predictions for validation data
        labels_predicted = model.predict(features_validation)

        # Return the binary cross-entropy
        return log_loss(labels_validation, labels_predicted)


def hyperopt_cv(
      params: dict = None,
      model = None,
      kfolds: int = 10,
      fold_split: float = 0.5,
      features: np.ndarray = None, 
      labels: np.ndarray = None
) -> float:
   
   '''Cross validate a classifier with a set of hyperparameters using 
   on-the-fly k-fold cross validation, returns mean CV score'''

   # Set the model parameters
   model.set_params(**params)

   # Get number of examples in dataset
   n = labels.shape[0]

   # Set score to zero at start
   score = 0

   # Run k-fold with random samples
   for k in range(kfolds):
      
      # Pick random indices without replacement for data to include in validation set
      validation_indices = np.random.choice(range(n), size = (int(n*fold_split),), replace = False)    
      validation_mask = np.zeros(n, dtype = bool)
      validation_mask[validation_indices] = True
      training_mask = ~validation_mask

      labels_train = labels[training_mask]
      features_train = features[training_mask]
      labels_validation = labels[validation_mask]
      features_validation = features[validation_mask]

      # Move data to GPU
      gpu_features_train = features_train
      gpu_labels_train = labels_train
      gpu_features_validation = features_validation
   
      # Fit the model
      model.fit(gpu_features_train, gpu_labels_train)

      # Make predictions for validation data
      labels_predicted = model.predict(gpu_features_validation)

      # Evaluate predictions, summing score across the folds
      score += log_loss(labels_validation, labels_predicted)

   # Return negated mean score for minimization
   return score / kfolds


def standard_scale_data(
        features_train_df: pd.DataFrame,
        features_test_df: pd.DataFrame,
        feature_column_names: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    '''Standard scales the training and testing data, returns
    pandas dataframes with column names preserved.'''

    # Instantiate standard scaler instance
    scaler = StandardScaler()

    # Fit on and transform training data
    scaled_features_train = scaler.fit_transform(features_train_df)

    # Use the scaler fit from the training data to transform the test data
    scaled_features_test = scaler.fit_transform(features_test_df)

    # Convert the scaled features back to pandas dataframe
    features_train_df = pd.DataFrame.from_records(scaled_features_train, columns = feature_column_names)
    features_test_df = pd.DataFrame.from_records(scaled_features_test, columns = feature_column_names)

    return features_train_df, features_test_df


def recursive_feature_elimination(
        features_train_df: pd.DataFrame,
        labels_train: np.ndarray, 
        cv_folds: int
) -> tuple[RFECV, pd.DataFrame, Callable]:
    
    '''Does recursive feature elimination with cross-validation'''

    rfecv = RFECV(
        estimator = XGBClassifier(),
        cv = KFold(cv_folds),
        scoring = make_scorer(negated_binary_cross_entropy),
        min_features_to_select = 1,
        step = 1,
        n_jobs = -1
    )

    rfecv.fit(features_train_df, labels_train)
    optimal_feature_count = rfecv.n_features_

    print(f'Optimal number of features: {optimal_feature_count}')

    cv_results = pd.DataFrame(rfecv.cv_results_)

    # Plot the data, if we have less than 20 feature sets, use a boxplot.
    # If we have more than that, use a scatter with error range
    if len(cv_results) <= 20:

        long_cv_results = pd.melt(cv_results.drop(['mean_test_score', 'std_test_score'], axis = 1).reset_index(), id_vars='index')
        long_cv_results.drop(['variable'], axis = 1, inplace = True)
        long_cv_results.rename({'index': 'n features', 'value': 'negated binary cross-entropy'}, axis = 1, inplace = True)
        long_cv_results['n features'] = long_cv_results['n features'] + 1

        sns.boxplot(y = 'negated binary cross-entropy', x = 'n features', data = long_cv_results)
        plt.title('Recursive Feature Elimination')

    else:

        plt.figure()
        plt.xlabel('Number of features selected')
        plt.ylabel('Mean test accuracy')
        plt.title('Recursive Feature Elimination')
    
        plt.fill_between(
            np.arange(len(cv_results['mean_test_score'])),
            cv_results['mean_test_score'] + cv_results['std_test_score'], # pylint: disable=line-too-long
            cv_results['mean_test_score'] - cv_results['std_test_score'], # pylint: disable=line-too-long
            alpha = 0.5
        )

        plt.plot(
            np.arange(len(cv_results['mean_test_score'])),
            cv_results['mean_test_score']
        )

    return rfecv, cv_results, plt


def sequential_feature_selection(
        features_train: pd.DataFrame,
        labels_train: pd.Series,
        feature_count: int,
        cv_folds: int
):
    
    '''Uses greedy sequential feature selection to choose n best features'''

    sfs = SequentialFeatureSelector(
        estimator = XGBClassifier(),
        cv = KFold(cv_folds),
        scoring = make_scorer(negated_binary_cross_entropy),
        n_features_to_select = feature_count,
        n_jobs = -1
    )

    fitted_sfs = sfs.fit(features_train, labels_train)

    return sfs, fitted_sfs


def sigma_clip_data(data: np.array, n_sigma: float = 5.0) -> np.array:
    '''Takes data and removes any points above or below the
    specified number of standard deviations away from the mean.'''

    # Get mean and standard deviation
    data_std_dev = np.std(data)
    data_mean = np.mean(data)

    # Remove points that more than n standard deviations below mean
    mask = data > data_mean - (data_std_dev * n_sigma)
    data = data[mask]

    # Remove data that are more than n standard deviations above the mean
    mask = data < data_mean + (data_std_dev * n_sigma)
    data = data[mask]

    return data


def make_padded_range(data: np.array, n_points: int = 100) -> np.array:
    '''Takes an input array and optionally a number of points. Generates
    a set of n_points sample points which span the data's range plus
    10% on either end. Returns the sample points.'''

    # Find the range of the data
    data_range = max(data) - min(data)

    # Calculate the padding amount as 10% of the data's range
    padding = data_range * 0.1

    # Get the padded min and max values
    x_max = max(data) + padding
    x_min = min(data) - padding

    # Determine the sampling frequency based on the caller
    # specified number of points
    sample_frequency = (x_max - x_min) / n_points

    # Make the points
    x = np.arange(x_min, x_max, sample_frequency)

    return x


def kde_bandwidth_scan(
        data: np.array,
        bandwidths: list
) -> dict:

    '''Takes 1D dataset and does gaussian KDE with SciPy and scikit-learn for a list of bandwidths'''

    # Pick a set of sample points which span the range of the input data to use for evaluation
    x = make_padded_range(data)

    # Empty dictionary to store results
    results = {}

    # Add empty lists for other values
    results['Package'] = []
    results['Bandwidth'] = []
    results['x'] = []
    results['y'] = []

    # Loop on the bandwidths to test
    for bandwidth in bandwidths:

        # Get KDEs for specified bandwidth
        scipy_kde = gaussian_kde(
            data.flatten(), 
            bw_method = bandwidth
        )

        scikit_kde = KernelDensity(
            kernel = 'gaussian',
            bandwidth = bandwidth
        ).fit(data.reshape(-1, 1))

        # Get the SciPy KDE's values at the sample points
        y = scipy_kde(x)
        
        # Clip data back to original range, and add to results
        for xi, yi in zip(x.flatten(), y.flatten()):
            if xi >= min(data) and xi <= max(data):
                results['Package'].append('SciPy')
                results['Bandwidth'].append(bandwidth)
                results['x'].append(xi)
                results['y'].append(yi)

        # Get the scikit KDE's values at the sample points
        log_y = scikit_kde.score_samples(x.reshape(-1, 1))
        y = np.exp(log_y)

        # Clip data back to original range, and add to results
        for xi, yi in zip(x.flatten(), y.flatten()):
            if xi >= min(data) and xi <= max(data):
                results['Package'].append('scikit-learn')
                results['Bandwidth'].append(bandwidth)
                results['x'].append(xi)
                results['y'].append(yi)

    results = pd.DataFrame(results)

    return results


def kde_speed(
    data: np.array,
    sample_sizes: list,
    replicates: int = 50
) -> dict:
    
    '''Takes a 1D array of data, and a list sample sizes. Does
    KDE with SciPy and scikit-learn for each sample size and 
    measures time. Returns results in a dictionary.'''

    # Empty dictionary to store results
    results = {}

    # Add empty lists for other values
    results['Replicate'] = []
    results['Package'] = []
    results['Sample size'] = []
    results['Mean time (sec.)'] = []
    results['Standard deviation'] = []

    # Loop on sample sizes
    for sample_size in sample_sizes:

        # Get the sample
        sample = np.random.choice(data.flatten(), size = sample_size)

        # Collectors for replicate data
        scipy_times = []
        scikit_times = []

        # Loop on replicates
        for i in range(replicates):

            # Do the KDEs, timing how long it takes

            # SciPy
            start_time = time.time()

            _ = gaussian_kde(
                sample,#.flatten(), 
                bw_method = 'silverman'
            )

            scipy_times.append(time.time() - start_time)

            # scikit-learn
            start_time = time.time()

            _ = KernelDensity(
                kernel = 'gaussian',
                bandwidth = 'silverman'
            ).fit(sample.reshape(-1, 1))

            scikit_times.append(time.time() - start_time)

        # Add results
        results['Replicate'].append(i)
        results['Package'].append('SciPy')
        results['Sample size'].append(sample_size)
        results['Mean time (sec.)'].append(np.mean(scipy_times))
        results['Standard deviation'].append(np.std(scipy_times))

        results['Replicate'].append(i)
        results['Package'].append('scikit-learn')
        results['Sample size'].append(sample_size)
        results['Mean time (sec.)'].append(np.mean(scikit_times))
        results['Standard deviation'].append(np.std(scikit_times))

    return results


def fitted_value_speed(
    data: np.array,
    n_eval_points: list,
    replicates: int = 5
) -> dict:
    
    '''Takes a 1D array of data, and a list evaluation point counts. Does
    KDE with SciPy and scikit-learn and then times the evaluation at each n 
    evaluation points, measures time. Returns results in a dictionary.'''

    # Do the KDEs

    # SciPy
    scipy_kde = gaussian_kde(
        data.flatten(), 
        bw_method = 'silverman'
    )

    # scikit-learn
    scikit_kde = KernelDensity(
        kernel = 'gaussian',
        bandwidth = 'silverman'
    ).fit(data.reshape(-1, 1))

    # Empty dictionary to store results
    results = {}

    # Add empty lists for other values
    results['Replicate'] = []
    results['Package'] = []
    results['Evaluation points'] = []
    results['Mean time (sec.)'] = []
    results['Standard deviation'] = []

    # Loop on evaluation point numbers 
    for n in n_eval_points:

        # Generate evaluation points
        x = make_padded_range(data, n)

        # Collectors for replicate data
        scipy_times = []
        scikit_times = []

        # Loop on replicates
        for i in range(replicates):

            # Do the evals, timing how long it takes

            # SciPy
            start_time = time.time()
            _ = scipy_kde(x)
            scipy_times.append(time.time() - start_time)

            # scikit-learn
            start_time = time.time()
            log_y = scikit_kde.score_samples(x.reshape(-1, 1))
            _ = np.exp(log_y)
            scikit_times.append(time.time() - start_time)

        # Add results
        results['Replicate'].append(i)
        results['Package'].append('SciPy')
        results['Evaluation points'].append(n)
        results['Mean time (sec.)'].append(np.mean(scipy_times))
        results['Standard deviation'].append(np.std(scipy_times))

        results['Replicate'].append(i)
        results['Package'].append('scikit-learn')
        results['Evaluation points'].append(n)
        results['Mean time (sec.)'].append(np.mean(scikit_times))
        results['Standard deviation'].append(np.std(scikit_times))

    return results


def parallel_score_samples(
        kde: gaussian_kde, 
        data: np.array, 
        workers: int
) -> np.ndarray:
    
    '''Splits evaluation over n_workers.'''

    with mp.Pool(workers) as p:
        return np.concatenate(p.map(kde, np.array_split(data, workers)))

def eval_speed_worker_count(
        data: np.array,
        worker_counts: list,
        replicates: int = 3
) -> dict:
    
    '''Tests evaluation speed with range of worker counts.'''

    # SciPy KDE
    scipy_kde = gaussian_kde(
        data.flatten(), 
        bw_method = 'silverman'
    )

    # Generate evaluation points
    x = make_padded_range(data, 16000)

    # Empty dictionary to store results
    results = {}

    # Add empty lists for other values
    results['Replicate'] = []
    results['Workers'] = []
    results['Mean time (sec.)'] = []
    results['Standard deviation'] = []

    # Loop worker counts 
    for worker_count in worker_counts:

        # Collector for replicate data
        times = []

        # Loop on replicates
        for i in range(replicates):

            # Do the eval, timing how long it takes
            start_time = time.time()
            _ = parallel_score_samples(
                kde = scipy_kde,
                data = x,
                workers = worker_count
            )
            times.append(time.time() - start_time)

        # Add results
        results['Replicate'].append(i)
        results['Workers'].append(worker_count)
        results['Mean time (sec.)'].append(np.mean(times))
        results['Standard deviation'].append(np.std(times))

    return results

################################################
# Experimental v2 inference pipeline functions #
################################################

def get_perplexity_ratio(input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    '''Takes text from input queue, calculates perplexity ratio and adds
    as new feature, puts updated input text into output queue.'''

    # Configure and load two instances of the model, one base for the
    # reader and one instruct for the writer. Use different GPUs.
    reader_model = llm_class.Llm(
        hf_model_string = config.READER_MODEL,
        device_map = config.READER_DEVICE
    )

    reader_model.load()
    print('Loaded reader model')

    writer_model = llm_class.Llm(
        hf_model_string = config.WRITER_MODEL,
        device_map = config.WRITER_DEVICE
    )

    writer_model.load()
    print('Loaded writer model')

    # Main loop
    while True:

        # Get text to score from input queue
        input_text=input_queue.get()

        # If we get the 'done' signal from the input queue
        # send it along and finish
        if input_text == 'done':
            output_queue.put('done')
            return

        # Start feature dictionary with new text
        features={'Text string': input_text}

        # Add the text length in words as the first feature
        features['Text length (words)']=len(input_text.split(' '))

        # Encode the string using the reader's tokenizer
        encodings = reader_model.tokenizer(
            input_text,
            return_tensors = 'pt',
            return_token_type_ids = False
        ).to(reader_model.device_map)

        # Get the string length in tokens and add to features
        fragment_length = encodings['input_ids'].shape[1]
        features['Text length (tokens)']=fragment_length

        # Calculate logits
        reader_logits = reader_model.model(**encodings).logits
        writer_logits = writer_model.model(**encodings).logits

        # Calculate perplexity and add to features
        ppl = perplexity(encodings, writer_logits)
        features['Perplexity']=ppl[0]

        # Calculate cross perplexity and add to features
        x_ppl = entropy(
            reader_logits.to(config.CALCULATION_DEVICE),
            writer_logits.to(config.CALCULATION_DEVICE),
            encodings.to(config.CALCULATION_DEVICE),
            reader_model.tokenizer.pad_token_id
        )

        features['Cross-perplexity']=x_ppl[0]

        # Calculate perplexity ratio and add to features
        scores = ppl / x_ppl
        scores = scores.tolist()
        perplexity_ratio_score = scores[0]
        features['Perplexity ratio']=perplexity_ratio_score

        # Put the text and the new features into the output queue
        output_queue.put(features)


def stage_one_classifier(input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    '''Does feature engineering and classification for stage I classifier.'''

    ###############################################################
    # Load the stage I feature engineering assets #################
    ###############################################################

    # Length bin definitions
    bins={
        'bin_001_050': [1, 50],
        'bin_026_075': [26, 75],
        'bin_051_100': [51, 100],
        'bin_076_125': [76, 125],
        'bin_101_150': [101, 150],
        'bin_126_175': [126, 175],
        'bin_151_200': [151, 200],
        'bin_176_225': [176, 225],
        'bin_201_250': [201, 250],
        'bin_226_275': [226, 275],
        'bin_251_300': [251, 300]
    }

    # Holders for perplexity ratio and TF-IDF score KLD kernel density estimates
    # and TF-IDF term frequency look-up tables for each text length bin
    perplexity_ratio_kld_kdes={}
    tfidf_kld_kdes={}
    tfidf_luts={}

    # Pre-load each kernel density estimate, using it's length bin as key
    for bin_id in bins.keys():

        # Load perplexity ratio KLD kernel density estimate for this bin
        perplexity_ratio_kld_kde_file=f'{config.MODELS_PATH}/perplexity_ratio_score_KLD_KDE_{bin_id}.pkl'
        with open(perplexity_ratio_kld_kde_file, 'rb') as input_file:
            perplexity_ratio_kld_kdes[bin_id]=pickle.load(input_file)

        # Load TF-IDF KLD kernel density estimate for this bin
        tfidf_kld_kde_file=f'{config.MODELS_PATH}/tf-idf_score_KLD_KDE_{bin_id}.pkl'
        with open(tfidf_kld_kde_file, 'rb') as input_file:
            tfidf_kld_kdes[bin_id]=pickle.load(input_file)

        # Load the TF-IDF look-up tables for this bin
        tfidf_luts_file=f'{config.MODELS_PATH}/tf-idf_luts_{bin_id}.pkl'
        with open(tfidf_luts_file, 'rb') as input_file:
            tfidf_luts[bin_id]=pickle.load(input_file)

    # Load the winning models for each length bin from the hyperparameter tuning results
    hyperparameter_iterations=100
    hyperparameter_optimization_results_filename=f'{config.DATA_PATH}/stage_one_hyperparameter_optimization_results_{hyperparameter_iterations}_iterations.pkl'
    with open(hyperparameter_optimization_results_filename, 'rb') as result_input_file:
        xgboost_models=pickle.load(result_input_file)

    ###############################################################
    # Main loop ###################################################
    ###############################################################

    while True:

        # Grab an input from the queue
        input_features=input_queue.get()

        # If we get the 'done' signal from the input queue
        # send it along and finish
        if input_features == 'done':
            output_queue.put('done')
            return
        
        ###############################################################
        # Load features and length bin the text #######################
        ###############################################################
        
        # Get the text, length and perplexity ratio from the input features
        text=input_features['Text string']
        text_length=input_features['Text length (words)']
        perplexity_ratio=input_features['Perplexity ratio']

        # Bin the fragment by length, we should get two matches here because of
        # the overlapping bins. Probably need to add some logic here to decide
        # what to do if we don't match on exactly two bins. Use the bin ids as
        # keys to start a dictionary of lists, we will accumulate feature vectors
        # for the classifier under each bin id
        text_features={}
        for bin_id, bin_range in bins.items():
            if text_length >= bin_range[0] and text_length <= bin_range[1]:
                text_features[bin_id]=[]

        # Add the features we already have: text length in tokens, perplexity,
        # cross-perplexity and perplexity-ratio score
        for bin_id in text_features.keys():
            text_features[bin_id].append(input_features['Text length (tokens)'])
            text_features[bin_id].append(input_features['Perplexity'])
            text_features[bin_id].append(input_features['Cross-perplexity'])
            text_features[bin_id].append(input_features['Perplexity ratio'])

        # # Pull the assets for the fragment's length bin
        # perplexity_ratio_kld_kde=perplexity_ratio_kld_kdes[text_bin_id]
        # tfidf_kld_kde=tfidf_kld_kdes[text_bin_id]
        # tfidf_lut=tfidf_luts[text_bin_id]
        # xgboost_model=xgboost_models[text_bin_id].best_estimator_

        ###############################################################
        # Perplexity ratio Kullback-Leibler score #####################
        ###############################################################

        # Calculate perplexity ratio KLD score for both bins and add to features
        for bin_id in text_features.keys():
            perplexity_ratio_kld_score=perplexity_ratio_kld_kdes[bin_id].pdf(perplexity_ratio)
            text_features[bin_id].append(perplexity_ratio_kld_score[0])

        ###############################################################
        # TF-IDF score ################################################
        ###############################################################

        # Clean the test for TF-IDF scoring
        sw=stopwords.words('english')
        lemmatizer=WordNetLemmatizer()

        cleaned_string=clean_text(
            text=text,
            sw=sw,
            lemmatizer=lemmatizer
        )

        # Split cleaned string into words
        words=cleaned_string.split(' ')

        # Calculate TF-IDF features for both bins
        for bin_id in text_features.keys():

            # Initialize TF-IDF sums
            human_tfidf_sum=0
            synthetic_tfidf_sum=0

            # Score the words using the human and synthetic luts
            for word in words:

                if word in tfidf_luts[bin_id]['human'].keys():
                    human_tfidf_sum += tfidf_luts[bin_id]['human'][word]

                if word in tfidf_luts[bin_id]['synthetic'].keys():
                    synthetic_tfidf_sum += tfidf_luts[bin_id]['synthetic'][word]

            # Get the means and add to features
            human_tfidf_mean=human_tfidf_sum / len(words)
            synthetic_tfidf_mean=synthetic_tfidf_sum / len(words)
            dmean_tfidf=human_tfidf_mean - synthetic_tfidf_mean
            product_normalized_dmean_tfidf=dmean_tfidf * (human_tfidf_mean + synthetic_tfidf_mean)

            text_features[bin_id].append(human_tfidf_mean)
            text_features[bin_id].append(synthetic_tfidf_mean)
            text_features[bin_id].append(product_normalized_dmean_tfidf)

        ###############################################################
        # TF-IDF Kullback-Leibler score ###############################
        ###############################################################

        # Calculate TF-IDF LKD score for each bin and add to features
        for bin_id in text_features.keys():
            tfidf_kld_score=tfidf_kld_kdes[bin_id].pdf(text_features[bin_id][-1])
            text_features[bin_id].append(tfidf_kld_score[0])

        ###############################################################
        # XGBoost classifier ##########################################
        ###############################################################
        
        # Run the classifier for each bin on the fragments
        for bin_id in text_features.keys():
            print(f'Running inference in {bin_id} with features: {text_features[bin_id]}')
            xgboost_model=xgboost_models[bin_id].best_estimator_
            class_probabilities=xgboost_model.predict_proba([text_features[bin_id]])

            # Get the class 0 probability
            class_probability=class_probabilities[0][0]

            # Add it to the original features
            input_features[f'Class probability {bin_id}']=class_probability

        # Done
        output_queue.put(input_features)


def stage_two_classifier(input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    '''Does feature engineering and classification for stage II classifier.'''

    while True:
        features=input_queue.get()

        # If we get the 'done' signal from the input queue
        # send it along and finish
        if features == 'done':
            output_queue.put('done')
            return
        
        output_queue.put(features)
    
# Take some care with '.sum(1)).detach().cpu().float().numpy()'. Had
# errors as cribbed from the above repo. Order matters? I don't know,
# current solution is very 'kitchen sink'

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction = 'none')
softmax_fn = torch.nn.Softmax(dim = -1)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0
):

    '''Perplexity score function'''

    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.detach().cpu().numpy()

    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0
):

    '''Cross entropy score function'''

    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(
        input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)
                   ).detach().cpu().float().numpy())

    return agg_ce

def clean_text(text: str = None, sw = None, lemmatizer = None) -> str:
    '''Cleans up text string for TF-IDF'''

    # Lowercase everything
    text = text.lower()

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", "",text)

    # Remove html tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'',text)

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

    # Remove punctuations
    for p in punctuations:
        text = text.replace(p,'')

    # Remove stopwords
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)

    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    return text