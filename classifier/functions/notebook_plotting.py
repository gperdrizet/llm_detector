'''Collection of functions for plotting in notebooks'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import functions.notebook_helper as helper_funcs
import functions.parallel_xgboost as xgb_funcs


def data_exploration_plot(data):
    '''Takes instance of class FeatureEngineering, makes some correlation
    and distribution diagnostic plots'''

    # Set up a 2 x 2 figure for some diagnostic plots
    _, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.3}
    )

    # Plot distribution of perplexity ratio scores
    axs[0,0].set_title('Perplexity ratio score distribution')
    axs[0,0].hist(data.all.human['Perplexity ratio score'], density = True, facecolor = 'green', label = 'Human text', alpha = 0.5)
    axs[0,0].hist(data.all.synthetic_combined['Perplexity ratio score'], density = True, facecolor = 'blue', label = 'Synthetic text', alpha = 0.5)
    axs[0,0].legend(loc = 'upper left')
    axs[0,0].set_xlabel('Perplexity ratio score')
    axs[0,0].set_ylabel('Fragments')

    # Scatter plot of perplexity vs cross-perplexity
    axs[0,1].set_title('Perplexity vs cross-perplexity')
    axs[0,1].scatter(data.all.human['Perplexity'], data.all.human['Cross-perplexity'], c = 'green', label = 'Human text')
    axs[0,1].scatter(data.all.synthetic_combined['Perplexity'], data.all.synthetic_combined['Cross-perplexity'], c = 'blue', label = 'Synthetic text')
    axs[0,1].legend(loc = 'lower right')
    axs[0,1].set_xlabel('Perplexity')
    axs[0,1].set_ylabel('Cross-perplexity')

    # Scatter plot of perplexity ratio score as a function of the
    # the text fragment length
    axs[1,0].set_title('Perplexity ratio score by fragment length')
    axs[1,0].scatter(data.all.human['Fragment length (words)'], data.all.human['Perplexity ratio score'], c = 'green', alpha = 0.5, label = 'Human text')
    axs[1,0].scatter(data.all.synthetic_combined['Fragment length (words)'], data.all.synthetic_combined['Perplexity ratio score'], c = 'blue', alpha = 0.5, label = 'Synthetic text')
    axs[1,0].legend(loc = 'upper right')
    axs[1,0].set_xlabel('Fragment length (words)')
    axs[1,0].set_ylabel('Perplexity ratio score')

    # Plot length distributions for human and synthetic text fragments
    axs[1,1].set_title('Fragment length distribution')
    axs[1,1].hist(data.all.human['Fragment length (words)'], density = True, facecolor = 'green', label = 'Human text', alpha = 0.5)
    axs[1,1].hist(data.all.synthetic_combined['Fragment length (words)'], density = True, facecolor = 'blue', label = 'Synthetic text', alpha = 0.5)
    axs[1,1].legend(loc = 'upper right')
    axs[1,1].set_xlabel('Fragment length (words)')
    axs[1,1].set_ylabel('Fragments')

    return plt


def data_exploration_plot_v2(data: pd.DataFrame, feature_name = str) -> plt:
    '''Takes dataframe and feature name, makes some correlation
    and distribution diagnostic plots'''

    # Separate human and synthetic data
    human_data = data[data['Source'] == 'human']
    synthetic_data = data[data['Source'] == 'synthetic']

    # Set up a 2 x 2 figure for some diagnostic plots
    _, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.3}
    )

    # Plot distribution of perplexity ratio scores
    axs[0,0].set_title(f'Perplexity ratio score distribution')
    axs[0,0].hist(human_data['Perplexity ratio score'], density = True, facecolor = 'green', label = 'Human text', alpha = 0.5)
    axs[0,0].hist(synthetic_data['Perplexity ratio score'], density = True, facecolor = 'blue', label = 'Synthetic text', alpha = 0.5)
    axs[0,0].legend(loc = 'upper left')
    axs[0,0].set_xlabel('Perplexity ratio score')
    axs[0,0].set_ylabel('Fragments')

    # Scatter plot of perplexity vs cross-perplexity
    axs[0,1].set_title('Perplexity vs cross-perplexity')
    axs[0,1].scatter(human_data['Perplexity'], human_data['Cross-perplexity'], c = 'green', label = 'Human text')
    axs[0,1].scatter(synthetic_data['Perplexity'], synthetic_data['Cross-perplexity'], c = 'blue', label = 'Synthetic text')
    axs[0,1].legend(loc = 'lower right')
    axs[0,1].set_xlabel('Perplexity')
    axs[0,1].set_ylabel('Cross-perplexity')

    # Scatter plot of perplexity ratio score as a function of the
    # the text fragment length
    axs[1,0].set_title('Perplexity ratio score by fragment length')
    axs[1,0].scatter(human_data['Fragment length (words)'], human_data['Perplexity ratio score'], c = 'green', alpha = 0.5, label = 'Human text')
    axs[1,0].scatter(synthetic_data['Fragment length (words)'], synthetic_data['Perplexity ratio score'], c = 'blue', alpha = 0.5, label = 'Synthetic text')
    axs[1,0].legend(loc = 'upper right')
    axs[1,0].set_xlabel('Fragment length (words)')
    axs[1,0].set_ylabel('Perplexity ratio score')

    # Plot length distributions for human and synthetic text fragments
    axs[1,1].set_title('Fragment length distribution')
    axs[1,1].hist(human_data['Fragment length (words)'], density = True, facecolor = 'green', label = 'Human text', alpha = 0.5)
    axs[1,1].hist(synthetic_data['Fragment length (words)'], density = True, facecolor = 'blue', label = 'Synthetic text', alpha = 0.5)
    axs[1,1].legend(loc = 'upper right')
    axs[1,1].set_xlabel('Fragment length (words)')
    axs[1,1].set_ylabel('Fragments')

    return plt


def perplexity_ratio_by_dataset(data):
    '''Creates boxplot of perplexity ratio score for human and synthetic
    text fragments separated by original source dataset.'''

    ax = sns.boxplot(data = data.all.combined, x = 'Dataset', y = 'Perplexity ratio score', hue = 'Source')
    ax.tick_params(axis = 'x', labelrotation = 45)

    return plt


def perplexity_ratio_by_length(data):
    '''Creates boxplot of perplexity ratio score for human and synthetic
    text fragments separated into bins by length'''

    # Bin the data
    binned_data = data.all.combined[['Fragment length (tokens)', 'Perplexity ratio score', 'Source']].copy()
    bins = pd.cut(binned_data.loc[:, 'Fragment length (tokens)'], 10)
    binned_data.loc[:, 'Length bin (tokens)'] = bins

    ax = sns.boxplot(data = binned_data, x = 'Length bin (tokens)', y = 'Perplexity ratio score', hue = 'Source')
    ax.tick_params(axis = 'x', labelrotation = 45)
    
    return plt


def plot_score_distribution_fits(
        plot_title,
        bin_centers,
        human_density,
        human_exp_gaussian_values,
        human_kde_values,
        synthetic_density,
        synthetic_exp_gaussian_values,
        synthetic_kde_values
):
    '''Plot score density and fits for human and synthetic data'''

    plt.scatter(bin_centers, human_density, label = 'human data', color = 'tab:blue', s = 10)
    plt.plot(bin_centers, human_exp_gaussian_values, label = 'human exp. gaussian', color = 'tab:blue')
    plt.plot(bin_centers, human_kde_values, label = 'human KDE', linestyle = 'dashed', color = 'tab:blue')
    plt.scatter(bin_centers, synthetic_density, label = 'synthetic data', color = 'tab:orange', s = 10)
    plt.plot(bin_centers, synthetic_exp_gaussian_values, label = 'synthetic exp. gaussian', color = 'tab:orange')
    plt.plot(bin_centers, synthetic_kde_values, label = 'human KDE', linestyle = 'dashed', color = 'tab:orange')

    plt.title(plot_title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend(loc = 'best', fontsize = 'small', markerscale = 2)

    return plt


def plot_fit_diagnostics(
        figure_title,
        bin_centers,
        human_density,
        synthetic_density,
        human_exp_gaussian_values,
        synthetic_exp_gaussian_values,
        human_kde_values,
        synthetic_kde_values
):
    '''Plot score density fit diagnostics'''

    fig, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.3},
        sharex='row',
        sharey='row'
    )

    fig.suptitle(figure_title, fontsize='x-large')

    axs[0,0].set_title('True value vs exponential\ngaussian fit')
    axs[0,0].scatter(human_density, human_exp_gaussian_values, label = 'human', s = 10)
    axs[0,0].scatter(synthetic_density, synthetic_exp_gaussian_values, label = 'synthetic', s = 10)
    axs[0,0].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[0,0].set_xlabel('True value')
    axs[0,0].set_ylabel('Fit value')

    axs[0,1].set_title('True value vs gaussian\nkernel density estimate')
    axs[0,1].scatter(human_density, human_kde_values, label = 'human', s = 10)
    axs[0,1].scatter(synthetic_density, synthetic_kde_values, label = 'synthetic', s = 10)
    axs[0,1].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[0,1].set_xlabel('True value')
    axs[0,1].set_ylabel('Fit value')

    axs[1,0].set_title('Exponential gaussian fit residuals')
    axs[1,0].scatter(bin_centers, human_density - human_exp_gaussian_values, label = 'human exp. gaussian', s = 10)
    axs[1,0].scatter(bin_centers, synthetic_density - synthetic_exp_gaussian_values, label = 'synthetic exp. gaussian', s = 10)
    axs[1,0].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[1,0].set_xlabel('Score')
    axs[1,0].set_ylabel('True - fitted value')

    axs[1,1].set_title('Gaussian kernel density estimate residuals')
    axs[1,1].scatter(bin_centers, human_density - human_kde_values, label = 'human KDE', s = 10)
    axs[1,1].scatter(bin_centers, synthetic_density - synthetic_kde_values, label = 'synthetic KDE', s = 10)
    axs[1,1].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[1,1].set_xlabel('Score')
    axs[1,1].set_ylabel('True - fitted value')

    return plt


def plot_kl_divergences(
        plot_title,
        bin_centers,
        human_exp_gaussian_values,
        synthetic_exp_gaussian_values,
        human_kde_values,
        synthetic_kde_values

):
    '''Plot Kullback-Leibler divergences'''
    
    fig, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.4},
        sharex='row',
        sharey='row'
    )

    fig.suptitle(plot_title, fontsize='x-large')

    axs[0,0].set_title('Exponential gaussian fits:\nsynthetic-human')
    axs[0,0].plot(bin_centers, human_exp_gaussian_values, label = 'human')
    axs[0,0].plot(bin_centers, synthetic_exp_gaussian_values, label = 'synthetic')
    axs[0,0].plot(bin_centers, helper_funcs.kl_divergence(synthetic_exp_gaussian_values, human_exp_gaussian_values), label = 'KL divergence')
    axs[0,0].set_xlabel('Score')
    axs[0,0].set_ylabel('Density')
    axs[0,0].legend(loc = 'upper right', fontsize = 'small')

    axs[0,1].set_title('Exponential gaussian fits:\nhuman-synthetic')
    axs[0,1].plot(bin_centers, human_exp_gaussian_values, label = 'human')
    axs[0,1].plot(bin_centers, synthetic_exp_gaussian_values, label = 'synthetic')
    axs[0,1].plot(bin_centers, helper_funcs.kl_divergence(human_exp_gaussian_values, synthetic_exp_gaussian_values), label = 'KL divergence')
    axs[0,1].set_xlabel('Score')
    axs[0,1].set_ylabel('Density')
    axs[0,1].legend(loc = 'upper left', fontsize = 'small')

    axs[1,0].set_title('Gaussian kernel density estimates:\nsynthetic-human')
    axs[1,0].plot(bin_centers, human_kde_values, label = 'human')
    axs[1,0].plot(bin_centers, synthetic_kde_values, label = 'synthetic')
    axs[1,0].plot(bin_centers, helper_funcs.kl_divergence(synthetic_kde_values, human_kde_values), label = 'KL divergence')
    axs[1,0].set_xlabel('Score')
    axs[1,0].set_ylabel('Density')
    axs[1,0].legend(loc = 'upper right', fontsize = 'small')

    axs[1,1].set_title('Gaussian kernel density estimates:\nhuman-synthetic')
    axs[1,1].plot(bin_centers, human_kde_values, label = 'human')
    axs[1,1].plot(bin_centers, synthetic_kde_values, label = 'synthetic')
    axs[1,1].plot(bin_centers, helper_funcs.kl_divergence(human_kde_values, synthetic_kde_values), label = 'KL divergence')
    axs[1,1].set_xlabel('Score')
    axs[1,1].set_ylabel('Density')
    axs[1,1].legend(loc = 'upper left', fontsize = 'small')

    return plt

def plot_cross_correlation_matrix(features_df):
    '''Plots cross correlation matrix for features dataframe.'''

    plt.matshow(features_df.corr())
    plt.xticks(range(features_df.select_dtypes(['number']).shape[1]), fontsize = 10)
    plt.yticks(range(features_df.select_dtypes(['number']).shape[1]), fontsize = 10)
    cb = plt.colorbar()
    _ = cb.ax.tick_params(labelsize = 12)
    plt.title('Feature cross correlation matrix')

    return plt


def plot_feature_distributions(features_df):
    '''Plots density distribution for each feature in dataframe.'''

    plot_titles = list(features_df.columns)
    plot_titles = ['Synthetic-human\nperplexity ratio\nexponential gaussian fit\nKullback-Leibler score' if x=='Synthetic-human perplexity ratio exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Human-synthetic\nperplexity ratio\nexponential gaussian fit\nKullback-Leibler score' if x=='Human-synthetic perplexity ratio exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Synthetic-human\nperplexity ratio\nkernel density estimate\nKullback-Leibler score' if x=='Synthetic-human perplexity ratio kernel density estimate Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Human-synthetic\nperplexity ratio\nkernel density estimate\nKullback-Leibler score' if x=='Human-synthetic perplexity ratio kernel density estimate Kullback-Leibler score' else x for x in plot_titles]

    plot_titles = ['Synthetic-human\nTF-IDF\nexponential gaussian fit\nKullback-Leibler score' if x=='Synthetic-human TF-IDF exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Human-synthetic\nTF-IDF\nexponential gaussian fit\nKullback-Leibler score' if x=='Human-synthetic TF-IDF exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Synthetic-human\nTF-IDF\nkernel density estimate\nKullback-Leibler score' if x=='Synthetic-human TF-IDF kernel density estimate Kullback-Leibler score' else x for x in plot_titles]
    plot_titles = ['Human-synthetic\nTF-IDF\nkernel density estimate\nKullback-Leibler score' if x=='Human-synthetic TF-IDF kernel density estimate Kullback-Leibler score' else x for x in plot_titles]

    plot_titles = ['Perplexity\nratio score' if x=='Perplexity ratio score' else x for x in plot_titles]
    plot_titles = ['Fragment length\n(words)' if x=='Fragment length (words)' else x for x in plot_titles]
    plot_titles = ['Fragment length\n(tokens)' if x=='Fragment length (tokens)' else x for x in plot_titles]

    n_cols = len(features_df.columns) // 4
    n_rows = (len(features_df.columns) // 4) + (len(features_df.columns) % 4)

    features_df.plot(title = plot_titles, kind = 'density', subplots = True, sharex = False, legend = False, layout = (n_rows,n_cols), figsize = (10,10))

    plt.tight_layout()

    return plt


def plot_scatter_matrix(features_df):
    '''Plots scatter matrix of features in dataframe'''

    axes = scatter_matrix(features_df, figsize = (10, 10), diagonal = 'kde')

    for ax in axes.flatten():

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace = 0, hspace = 0)
    
    return plt


def plot_cross_validation(plots, results):
    '''Takes a list of independent variables and the results dictionary,
    makes and returns boxplots.'''

    # Set-up the subplots
    num_conditions = len(set(results['Condition']))
    _, axes = plt.subplots(5, 1, figsize=(7, num_conditions + 3))

    # Draw each boxplot
    for plot, ax in zip(plots, axes.flatten()):
        sns.boxplot(y = 'Condition', x = plot, data = pd.DataFrame.from_dict(results), orient = 'h', ax = ax)
        
    plt.tight_layout()

    return plt


def plot_two_factor_cross_validation(plots, results):
    '''Takes a list of independent variables and the results dictionary,
    makes and returns boxplots.'''

    # Set-up the subplots
    num_conditions = len(set(results['Condition']))
    _, axes = plt.subplots(4, 1, figsize=(7, num_conditions + 1))

    # Draw each boxplot
    for plot, ax in zip(plots, axes.flatten()):
        sns.boxplot(y = 'Condition', x = plot, hue = 'Optimized', data = pd.DataFrame.from_dict(results), orient = 'h', ax = ax)
        
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


def plot_hyperparameter_tuning(cv_results: pd.DataFrame) -> plt:
    '''Takes parsed results from parse_hyperparameter_tuning_results()
    in parallel_xgboost functions. plots the binary cross entropy 
    of each step by the step's rank for each bin, returns the plot object'''

    # Set up a figure for 12 bins
    _, axs = plt.subplots(
        4,
        3,
        figsize = (9, 12),
        gridspec_kw = {'wspace':0.4, 'hspace':0.4}
    )

    # Plot the results for each bin on a separate axis
    for bin_id, ax in zip(cv_results['bin'].unique(), axs.reshape(-1)):

        bin_results = cv_results[cv_results['bin'] == bin_id]
        sorted_bin_results = bin_results.sort_values('rank_test_negated_binary_cross_entropy')

        ax.set_title(f'{bin_id}')
        ax.set_xlabel('Parameter set rank')
        ax.set_ylabel('Binary cross-entropy')
        ax.invert_xaxis()

        ax.fill_between(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_test_binary_cross_entropy'] + sorted_bin_results['std_test_binary_cross_entropy'],
            sorted_bin_results['mean_test_binary_cross_entropy'] - sorted_bin_results['std_test_binary_cross_entropy'],
            alpha = 0.5
        )

        ax.plot(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_test_binary_cross_entropy'],
            label='Validation'
        )

        ax.fill_between(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_train_binary_cross_entropy'] + sorted_bin_results['std_train_binary_cross_entropy'],
            sorted_bin_results['mean_train_binary_cross_entropy'] - sorted_bin_results['std_train_binary_cross_entropy'],
            alpha = 0.5
        )

        ax.plot(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_train_binary_cross_entropy'],
            label='Training'
        )

        ax.legend(loc = 'best', fontsize = 'x-small')

    return plt


def plot_testing_confusion_matrices(winners: dict, input_file: str) -> plt:
    '''Takes winners dictionary from parse_hyperparameter_tuning_results()
    in parallel_xgboost functions and path to hdf5 datafile. Plots confusion 
    matrix using predictions on hold-out test data for each bin.'''

    # Will need to get the test data for each bin, open a 
    # connection to the hdf5 dataset via PyTables with Pandas
    data_lake = pd.HDFStore(input_file)

    # Set up a figure for 12 bins
    _, axs = plt.subplots(
        6,
        2,
        figsize = (8, 18),
        gridspec_kw = {'hspace':0.5}
    )

    # Now, loop on the winners and the subplots
    for bin_id, ax in zip(winners.keys(), axs.reshape(-1)):

        # Get the model
        model = winners[bin_id]['model']

        # Get the testing features and labels
        features_df = data_lake[f'testing/{bin_id}/features']
        labels = data_lake[f'testing/{bin_id}/labels']

        # Clean up the features
        features = xgb_funcs.prep_data(features_df, False)

        print(features.info())

        # Make the confusion matrix
        _ = ConfusionMatrixDisplay.from_estimator(
            model,
            features, 
            labels,
            display_labels = ['human', 'synthetic'],
            normalize = 'all',
            ax = ax
        )

        ax.title.set_text(bin_id)

    data_lake.close()

    return plt