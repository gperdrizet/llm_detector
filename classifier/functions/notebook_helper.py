'''Collection of functions refactored from notebooks for data handling'''

import re
import nltk
import cupy as cp
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, SplineTransformer

from math import log2
from statistics import mean
from scipy.stats import ttest_ind, fit, exponnorm, gaussian_kde
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def exp_gaussian_fit(scores):
    '''Fit and return exponnorm on score'''

    # Set parameter bounds
    bounds = [[0.001, 1.0], [0.001, 1.0], [0.001, 1.0]]

    # Do the fit
    exponnorm_fit = fit(exponnorm, scores.astype(np.float64), bounds = bounds)

    # Build function from rit
    exponnorm_func = exponnorm(exponnorm_fit.params.K, exponnorm_fit.params.loc, exponnorm_fit.params.scale)

    # Print the fitted parameters
    print(f'  Rate: {exponnorm_fit.params.K}')
    print(f'  Mean: {exponnorm_fit.params.loc}')
    print(f'  Variance: {exponnorm_fit.params.scale}')

    # Return the function
    return exponnorm_func


def kl_divergence(p, q):
    '''Takes two lists, calculates KD divergence'''

    results = []

    for i, j in zip(p, q):
        if i > 0 and j > 0:
            results.append(i * log2(i/j))

        else:
            results.append(np.nan)

    return np.asarray(results)


def get_kl_kde(figure_title, scores, human_fit_func, synthetic_fit_func, padding, sample_frequency):
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

    print(type(synthetic_human_kld))

    # Get rid of any np.nan
    # synthetic_human_kld = synthetic_human_kld[~np.isnan(synthetic_human_kld)]
    # human_synthetic_kld = human_synthetic_kld[~np.isnan(human_synthetic_kld)]
    mask = np.isnan(synthetic_human_kld)
    synthetic_human_kld[mask] = 0

    mask = np.isnan(human_synthetic_kld)
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

    plt.show()

    return synthetic_human_kld_kde, human_synthetic_kld_kde, plt


nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def clean_text(text: str = None) -> str:
    
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


def prep_training_data(data):
    '''Takes instance of feature engineering class, prepares data for classifier training'''

    # Retrieve training and testing data
    training_data_df = data.training.all.combined.copy()
    testing_data_df = data.testing.all.combined.copy()

    # Remove rows containing NAN
    training_data_df.dropna(inplace = True)
    testing_data_df.dropna(inplace = True)

    # Drop unnecessary or un-trainable features
    feature_drops = [
        'Source record num',
        'Dataset',
        'Generator',
        'String',
        'Reader time (seconds)',
        'Writer time (seconds)',
        'Reader peak memory (GB)',
        'Writer peak memory (GB)'
    ]

    training_data_df.drop(feature_drops, axis = 1, inplace = True)
    testing_data_df.drop(feature_drops, axis = 1, inplace = True)

    # Split the data into features and labels
    labels_train = training_data_df['Source']
    features_train_df = training_data_df.drop('Source', axis = 1)

    labels_test = testing_data_df['Source']
    features_test_df = testing_data_df.drop('Source', axis = 1)

    # Encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(labels_train)
    labels_train = label_encoder.transform(labels_train)
    labels_test = label_encoder.transform(labels_test)

    print(f'Training data: {len(labels_train)} examples')
    print(f'Test data: {len(labels_test)} examples')

    return features_train_df, features_test_df, labels_train, labels_test


def standard_scale_data(features_train_df, features_test_df, feature_column_names):
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


def add_poly_features(features_train_df, features_test_df):
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

def add_spline_features(features_train_df, features_test_df):
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


def recursive_feature_elimination(features_train_df, labels_train, cv_folds, n_jobs):
    '''Does recursive feature elimination with cross-validation'''

    rfecv = RFECV(
        estimator = XGBClassifier(),
        cv = KFold(cv_folds),
        scoring = make_scorer(negated_binary_cross_entropy),
        min_features_to_select = 1,
        step = 1,
        n_jobs = n_jobs
    )

    rfecv.fit(features_train_df, labels_train)
    optimal_feature_count = rfecv.n_features_

    print(f'Optimal number of features: {optimal_feature_count}')

    cv_results = pd.DataFrame(rfecv.cv_results_)

    # Plot the data, if we have less than 20 feature sets, use a boxplot.
    # If we have more than that, use a scatter with error bars

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
        plt.errorbar(
            x=np.arange(len(cv_results['mean_test_score'])),
            y=cv_results['mean_test_score'],
            yerr=cv_results['std_test_score'],
        )
        plt.title('Recursive Feature Elimination')

    return rfecv, cv_results, plt

def sequential_feature_selection(features_train, labels_train, feature_count, cv_folds, n_jobs):
    '''Uses greedy sequential feature selection to choose n best features'''

    sfs = SequentialFeatureSelector(
        estimator = XGBClassifier(),
        cv = KFold(cv_folds),
        scoring = make_scorer(negated_binary_cross_entropy),
        n_features_to_select = feature_count,
        n_jobs = n_jobs
    )

    fitted_sfs = sfs.fit(features_train, labels_train)

    return sfs, fitted_sfs


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

def negated_binary_cross_entropy(labels, predictions):
    '''Scoring function for use with scikit-learn make_scorer
    takes a model, features and labels. Returns negated log loss 
    for binary classification. For use in situations where 
    'larger-is-better is desirable'''

    # Get the scikit-learn normalized accuracy score
    log_loss_score = log_loss(labels, predictions, normalize = True)

    return -log_loss_score


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