'''Collection of functions refactored from notebooks for data handling'''

import cupy as cp
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

from statistics import mean
from scipy.stats import ttest_ind
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# Set cupy CUDA device to GPU 0 (this is the GTX1070 on pyrite)
cp.cuda.Device(0).use()

def mean_difference_ci(data):
    '''Conducts t-test on difference in perplexity ratio score means on
    length binned data. Prints, means, difference in means, 95% confidence 
    interval around the difference in means and the p-value for the
    difference in means.
    '''

    # Bin the data
    binned_data = data.all.combined[['Fragment length (tokens)', 'Perplexity ratio score', 'Source']].copy()
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

def add_cv_scores(results, scores, condition):
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


def percent_accuracy(labels, predictions):
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns mean accuracy
    of predictions as a percent.'''

    # Get the scikit-learn normalized accuracy score
    accuracy = accuracy_score(labels, predictions, normalize = True) * 100

    return accuracy


def binary_cross_entropy(labels, predictions):
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns non-negated
    log loss for binary classification.'''

    # Get the scikit-learn normalized accuracy score
    log_loss_score = log_loss(labels, predictions, normalize = True)

    return log_loss_score


def false_positive_rate(labels, predictions):
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. returns false positive
    rate from scikit-learn confusion matrix.'''

    # Extract counts from confusion matrix
    _, fp, _, tp = confusion_matrix(labels, predictions).ravel()

    # Calculate the false positive rate
    false_positive_rate = fp / (tp + fp)

    return false_positive_rate


def false_negative_rate(labels, predictions):
    '''Scoring function for use with scikit-learn make_scorer
    take
    return false_positive_rates a model, features and labels. Returns false negative
    rate from scikit-learn confusion matrix.'''

    # Extract counts from confusion matrix
    tn, _, fn, _ = confusion_matrix(labels, predictions).ravel()

    # Calculate the false positive rate
    false_positive_rate = fn / (tn + fn)

    return false_positive_rate


def hyperopt(
      params: dict = None,
      model = None,
      features_training: np.ndarray = None,
      labels_training: np.ndarray = None,
      features_validation: np.ndarray = None,
      labels_validation: np.ndarray = None
) -> float:
   
    '''Cross validate a HistGradientBoostingClassifier classifier with a set of hyperparameters, 
    using a single, static training/validation split and early stopping. Return scores.'''

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
   
   '''Cross validate an XGBoost classifier with a set of hyperparameters using 
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


#################################################################################
# Plotting functions

def plot_cross_validation(plots, results):
    '''Takes a list of independent variables and the results dictionary,
    makes and returns boxplots.'''

    # Set-up the subplots
    fig, axes = plt.subplots(5, 1, figsize=(7, 7))

    # Draw each boxplot
    for plot, ax in zip(plots, axes.flatten()):
        sns.boxplot(y = 'Condition', x = plot, data = pd.DataFrame.from_dict(results), orient = 'h', ax = ax)
        
    plt.tight_layout()

    return plt


def make_optimization_plot(trials):
    '''Parse optimization trial results, make and return plot.'''

    # Find the parameter names
    parameters = list(trials[0]['misc']['vals'].keys())
    column_names = ['loss'] + parameters

    # Make and empty dictionary to hold the parsed results
    plot_data = {}

    # Add the column names as keys with empty list as value
    for column_name in column_names:
        plot_data[column_name] = []

    # Loop on the optimization trials
    for trial in trials:

        # Loop on the column names
        for column_name in column_names:

            # Grab the loss
            if column_name == 'loss':
                plot_data['loss'].append(trial['result']['loss'])

            # Grab the parameters
            else:
                plot_data[column_name].append(trial['misc']['vals'][column_name][0])

    # Convert to dataframe for plotting
    plot_data_df = pd.DataFrame.from_dict(plot_data)

    # Draw the plot
    optimization_plot = plot_data_df.plot(subplots = True, figsize = (12, 8))

    return optimization_plot