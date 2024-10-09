'''Collection of functions for plotting in notebooks'''

import numpy as np
import pandas as pd
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import functions.notebook_helper as helper_funcs
import functions.length_binned_xgboost as xgb_funcs


# def data_exploration_plot(data):
#     '''Takes instance of class FeatureEngineering, makes some correlation
#     and distribution diagnostic plots'''

#     # Set up a 2 x 2 figure for some diagnostic plots
#     _, axs = plt.subplots(
#         2,
#         2,
#         figsize = (8, 8),
#         gridspec_kw = {'wspace':0.3, 'hspace':0.3}
#     )

#     # Plot distribution of perplexity ratio scores
#     axs[0,0].set_title('Perplexity ratio score distribution')

#     axs[0,0].hist(
#         data.all.human['Perplexity ratio score'],
#         density = True,
#         facecolor = 'green',
#         label = 'Human text',
#         alpha = 0.5
#     )

#     axs[0,0].hist(
#         data.all.synthetic_combined['Perplexity ratio score'],
#         density = True,
#         facecolor = 'blue',
#         label = 'Synthetic text',
#         alpha = 0.5
#     )

#     axs[0,0].legend(loc = 'upper left')
#     axs[0,0].set_xlabel('Perplexity ratio score')
#     axs[0,0].set_ylabel('Fragments')

#     # Scatter plot of perplexity vs cross-perplexity
#     axs[0,1].set_title('Perplexity vs cross-perplexity')

#     axs[0,1].scatter(
#         data.all.human['Perplexity'],
#         data.all.human['Cross-perplexity'],
#         c = 'green',
#         label = 'Human text'
#     )

#     axs[0,1].scatter(
#         data.all.synthetic_combined['Perplexity'],
#         data.all.synthetic_combined['Cross-perplexity'],
#         c = 'blue',
#         label = 'Synthetic text'
#     )

#     axs[0,1].legend(loc = 'lower right')
#     axs[0,1].set_xlabel('Perplexity')
#     axs[0,1].set_ylabel('Cross-perplexity')

#     # Scatter plot of perplexity ratio score as a function of the
#     # the text fragment length
#     axs[1,0].set_title('Perplexity ratio score by fragment length')

#     axs[1,0].scatter(
#         data.all.human['Fragment length (words)'],
#         data.all.human['Perplexity ratio score'],
#         c = 'green',
#         alpha = 0.5,
#         label = 'Human text'
#     )

#     axs[1,0].scatter(
#         data.all.synthetic_combined['Fragment length (words)'],
#         data.all.synthetic_combined['Perplexity ratio score'],
#         c = 'blue',
#         alpha = 0.5,
#         label = 'Synthetic text'
#     )

#     axs[1,0].legend(loc = 'upper right')
#     axs[1,0].set_xlabel('Fragment length (words)')
#     axs[1,0].set_ylabel('Perplexity ratio score')

#     # Plot length distributions for human and synthetic text fragments
#     axs[1,1].set_title('Fragment length distribution')

#     axs[1,1].hist(
#         data.all.human['Fragment length (words)'],
#         density = True,
#         facecolor = 'green',
#         label = 'Human text',
#         alpha = 0.5
#     )

#     axs[1,1].hist(
#         data.all.synthetic_combined['Fragment length (words)'],
#         density = True,
#         facecolor = 'blue',
#         label = 'Synthetic text',
#         alpha = 0.5
#     )

#     axs[1,1].legend(loc = 'upper right')
#     axs[1,1].set_xlabel('Fragment length (words)')
#     axs[1,1].set_ylabel('Fragments')

#     return plt


def data_exploration_plot(data_df: pd.DataFrame) -> plt:
    '''Takes dataframe makes some correlation
    and distribution diagnostic plots'''

    # Separate human and synthetic data
    human_data = data_df[data_df['Source'] == 'human']
    synthetic_data = data_df[data_df['Source'] == 'synthetic']

    # Set up a 2 x 2 figure for some diagnostic plots
    _, axs = plt.subplots(
        2,
        2,
        figsize = (8, 8),
        gridspec_kw = {'wspace':0.3, 'hspace':0.3}
    )

    # Plot distribution of perplexity ratio scores
    axs[0,0].set_title('Perplexity ratio score distribution')

    axs[0,0].hist(
        human_data['Perplexity ratio score'],
        density = True,
        facecolor = 'green',
        label = 'Human text',
        alpha = 0.5
    )

    axs[0,0].hist(
        synthetic_data['Perplexity ratio score'],
        density = True,
        facecolor = 'blue',
        label = 'Synthetic text',
        alpha = 0.5
    )

    axs[0,0].legend(loc = 'upper left')
    axs[0,0].set_xlabel('Perplexity ratio score')
    axs[0,0].set_ylabel('Fragments')

    # Scatter plot of perplexity vs cross-perplexity
    axs[0,1].set_title('Perplexity vs cross-perplexity')
    axs[0,1].scatter(
        human_data['Perplexity'],
        human_data['Cross-perplexity'],
        c = 'green',
        label = 'Human text'
    )

    axs[0,1].scatter(
        synthetic_data['Perplexity'],
        synthetic_data['Cross-perplexity'],
        c = 'blue',
        label = 'Synthetic text'
    )

    axs[0,1].legend(loc = 'lower right')
    axs[0,1].set_xlabel('Perplexity')
    axs[0,1].set_ylabel('Cross-perplexity')

    # Scatter plot of perplexity ratio score as a function of the
    # the text fragment length
    axs[1,0].set_title('Perplexity ratio score by fragment length')
    axs[1,0].scatter(
        human_data['Fragment length (words)'],
        human_data['Perplexity ratio score'],
        c = 'green',
        alpha = 0.5,
        label = 'Human text'
    )

    axs[1,0].scatter(
        synthetic_data['Fragment length (words)'],
        synthetic_data['Perplexity ratio score'],
        c = 'blue',
        alpha = 0.5,
        label = 'Synthetic text'
    )

    axs[1,0].legend(loc = 'upper right')
    axs[1,0].set_xlabel('Fragment length (words)')
    axs[1,0].set_ylabel('Perplexity ratio score')

    # Plot length distributions for human and synthetic text fragments
    axs[1,1].set_title('Fragment length distribution')
    axs[1,1].hist(
        human_data['Fragment length (words)'],
        density = True,
        facecolor = 'green',
        label = 'Human text',
        alpha = 0.5
    )

    axs[1,1].hist(
        synthetic_data['Fragment length (words)'],
        density = True,
        facecolor = 'blue',
        label = 'Synthetic text',
        alpha = 0.5
    )

    axs[1,1].legend(loc = 'upper right')
    axs[1,1].set_xlabel('Fragment length (words)')
    axs[1,1].set_ylabel('Fragments')

    return plt


def perplexity_ratio_by_dataset(data_df: pd.DataFrame) -> plt:
    '''Creates boxplot of perplexity ratio score for human and synthetic
    text fragments separated by original source dataset.'''

    ax = sns.boxplot(
        data = data_df,
        x = 'Dataset',
        y = 'Perplexity ratio score',
        hue = 'Source'
    )

    ax.legend(loc = 'upper right')

    return plt


def perplexity_ratio_by_length(data_df: pd.DataFrame) -> plt:
    '''Creates boxplot of perplexity ratio score for human and synthetic
    text fragments separated into bins by length'''

    # Bin the data
    binned_data = data_df[[
        'Fragment length (tokens)',
        'Perplexity ratio score',
        'Source'
    ]].copy()

    bins = pd.cut(binned_data.loc[:, 'Fragment length (tokens)'], 10)
    binned_data.loc[:, 'Length bin (tokens)'] = bins

    ax = sns.boxplot(
        data = binned_data,
        x = 'Length bin (tokens)',
        y = 'Perplexity ratio score',
        hue = 'Source'
    )

    ax.tick_params(axis = 'x', labelrotation = 45)

    return plt


def plot_score_distribution_fits(
        plot_title: str = None,
        bin_centers: np.ndarray = None,
        human_density: np.ndarray = None,
        human_exp_gaussian_values: np.ndarray = None,
        human_kde_values: np.ndarray = None,
        synthetic_density: np.ndarray = None,
        synthetic_exp_gaussian_values: np.ndarray = None,
        synthetic_kde_values: np.ndarray = None
) -> plt:
    
    '''Plot score density and fits for human and synthetic data'''

    plt.scatter(
        bin_centers,
        human_density,
        label = 'human data',
        color = 'tab:blue',
        s = 10
    )

    plt.plot(
        bin_centers,
        human_exp_gaussian_values,
        label = 'human exp. gaussian',
        color = 'tab:blue'
    )

    plt.plot(
        bin_centers,
        human_kde_values,
        label = 'human KDE',
        linestyle = 'dashed',
        color = 'tab:blue'
    )

    plt.scatter(
        bin_centers,
        synthetic_density,
        label = 'synthetic data',
        color = 'tab:orange',
        s = 10
    )

    plt.plot(
        bin_centers,
        synthetic_exp_gaussian_values,
        label = 'synthetic exp. gaussian',
        color = 'tab:orange'
    )

    plt.plot(
        bin_centers,
        synthetic_kde_values,
        label = 'human KDE',
        linestyle = 'dashed',
        color = 'tab:orange'
    )

    plt.title(plot_title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend(loc = 'best', fontsize = 'small', markerscale = 2)

    return plt


def plot_fit_diagnostics(
        figure_title: str = None,
        bin_centers: np.ndarray = None,
        human_density: np.ndarray = None,
        synthetic_density: np.ndarray = None,
        human_exp_gaussian_values: np.ndarray = None,
        synthetic_exp_gaussian_values: np.ndarray = None,
        human_kde_values: np.ndarray = None,
        synthetic_kde_values: np.ndarray = None
) -> plt:
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

    axs[1,0].scatter(
        bin_centers,
        human_density - human_exp_gaussian_values,
        label = 'human exp. gaussian',
        s = 10
    )

    axs[1,0].scatter(
        bin_centers,
        synthetic_density - synthetic_exp_gaussian_values,
        label = 'synthetic exp. gaussian',
        s = 10
    )

    axs[1,0].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[1,0].set_xlabel('Score')
    axs[1,0].set_ylabel('True - fitted value')

    axs[1,1].set_title('Gaussian kernel density estimate residuals')

    axs[1,1].scatter(
        bin_centers,
        human_density - human_kde_values,
        label = 'human KDE',
        s = 10
    )

    axs[1,1].scatter(
        bin_centers,
        synthetic_density - synthetic_kde_values,
        label = 'synthetic KDE',
        s = 10
    )

    axs[1,1].legend(loc = 'upper left', fontsize = 'small', markerscale = 2)
    axs[1,1].set_xlabel('Score')
    axs[1,1].set_ylabel('True - fitted value')

    return plt


def plot_kl_divergences(
        plot_title: str = None,
        bin_centers: np.ndarray = None,
        human_exp_gaussian_values: np.ndarray = None,
        synthetic_exp_gaussian_values: np.ndarray = None,
        human_kde_values: np.ndarray = None,
        synthetic_kde_values: np.ndarray = None
) -> plt:
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

    axs[0,0].plot(
        bin_centers,
        helper_funcs.kl_divergence(synthetic_exp_gaussian_values, human_exp_gaussian_values),
        label = 'KL divergence'
    )

    axs[0,0].set_xlabel('Score')
    axs[0,0].set_ylabel('Density')
    axs[0,0].legend(loc = 'upper right', fontsize = 'small')

    axs[0,1].set_title('Exponential gaussian fits:\nhuman-synthetic')
    axs[0,1].plot(bin_centers, human_exp_gaussian_values, label = 'human')
    axs[0,1].plot(bin_centers, synthetic_exp_gaussian_values, label = 'synthetic')

    axs[0,1].plot(
        bin_centers,
        helper_funcs.kl_divergence(human_exp_gaussian_values, synthetic_exp_gaussian_values),
        label = 'KL divergence'
    )

    axs[0,1].set_xlabel('Score')
    axs[0,1].set_ylabel('Density')
    axs[0,1].legend(loc = 'upper left', fontsize = 'small')

    axs[1,0].set_title('Gaussian kernel density estimates:\nsynthetic-human')
    axs[1,0].plot(bin_centers, human_kde_values, label = 'human')
    axs[1,0].plot(bin_centers, synthetic_kde_values, label = 'synthetic')

    axs[1,0].plot(
        bin_centers,
        helper_funcs.kl_divergence(synthetic_kde_values, human_kde_values),
        label = 'KL divergence'
    )

    axs[1,0].set_xlabel('Score')
    axs[1,0].set_ylabel('Density')
    axs[1,0].legend(loc = 'upper right', fontsize = 'small')

    axs[1,1].set_title('Gaussian kernel density estimates:\nhuman-synthetic')
    axs[1,1].plot(bin_centers, human_kde_values, label = 'human')
    axs[1,1].plot(bin_centers, synthetic_kde_values, label = 'synthetic')

    axs[1,1].plot(
        bin_centers,
        helper_funcs.kl_divergence(human_kde_values, synthetic_kde_values),
        label = 'KL divergence'
    )

    axs[1,1].set_xlabel('Score')
    axs[1,1].set_ylabel('Density')
    axs[1,1].legend(loc = 'upper left', fontsize = 'small')

    return plt


def plot_feature_distributions(features_df: pd.DataFrame) -> plt:
    '''Plots density distribution for each feature in dataframe.'''

    plot_titles = list(features_df.columns)

    plot_titles = ['Synthetic-human\nperplexity ratio\nexponential gaussian fit\nKullback-Leibler score' if x=='Synthetic-human perplexity ratio exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Human-synthetic\nperplexity ratio\nexponential gaussian fit\nKullback-Leibler score' if x=='Human-synthetic perplexity ratio exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Synthetic-human\nperplexity ratio\nkernel density estimate\nKullback-Leibler score' if x=='Synthetic-human perplexity ratio kernel density estimate Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Human-synthetic\nperplexity ratio\nkernel density estimate\nKullback-Leibler score' if x=='Human-synthetic perplexity ratio kernel density estimate Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long

    plot_titles = ['Synthetic-human\nTF-IDF\nexponential gaussian fit\nKullback-Leibler score' if x=='Synthetic-human TF-IDF exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Human-synthetic\nTF-IDF\nexponential gaussian fit\nKullback-Leibler score' if x=='Human-synthetic TF-IDF exponential gaussian fit Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Synthetic-human\nTF-IDF\nkernel density estimate\nKullback-Leibler score' if x=='Synthetic-human TF-IDF kernel density estimate Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Human-synthetic\nTF-IDF\nkernel density estimate\nKullback-Leibler score' if x=='Human-synthetic TF-IDF kernel density estimate Kullback-Leibler score' else x for x in plot_titles] # pylint: disable=line-too-long

    plot_titles = ['Perplexity\nratio score' if x=='Perplexity ratio score' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Fragment length\n(words)' if x=='Fragment length (words)' else x for x in plot_titles] # pylint: disable=line-too-long
    plot_titles = ['Fragment length\n(tokens)' if x=='Fragment length (tokens)' else x for x in plot_titles] # pylint: disable=line-too-long

    n_cols = 4
    n_rows = (len(features_df.columns) // 4) + (len(features_df.columns) % 4)

    features_df.plot(
        title = plot_titles,
        kind = 'density',
        subplots = True,
        sharex = False,
        legend = False,
        layout = (n_rows,n_cols),
        figsize = (9,n_rows*2)
    )

    plt.tight_layout()

    return plt


def plot_cross_correlation_matrix(features_df: pd.DataFrame) -> plt:
    '''Plots cross correlation matrix for features dataframe.'''

    plt.matshow(features_df.corr())
    plt.xticks(range(features_df.select_dtypes(['number']).shape[1]), fontsize = 10)
    plt.yticks(range(features_df.select_dtypes(['number']).shape[1]), fontsize = 10)
    cb = plt.colorbar(shrink = 0.8)
    _ = cb.ax.tick_params(labelsize = 12)
    plt.title('Feature cross correlation matrix')

    return plt


def plot_scatter_matrix(features_df:pd.DataFrame) -> plt:
    '''Plots scatter matrix of features in dataframe'''

    # Get the cross correlation matrix
    correlations = features_df.corr(method = 'spearman')
    correlations = correlations.to_numpy().flatten()

    # Get the feature names from the dataframe
    num_features = len(features_df.columns)
    print(f'Have {num_features} features for plot:')

    # Number and rename the features for plotting
    feature_nums = {}

    for i, feature in enumerate(features_df.columns):
        print(f' {i}: {feature}')
        feature_nums[feature] = i

    features_df.rename(feature_nums, axis = 1, inplace = True)

    # Get feature number pairs for each plot using the cartesian product of the feature numbers
    feature_pairs = itertools.product(list(range(num_features)), list(range(num_features)))
    
    # Assign fraction of plot width to scatter matrix and colorbar
    colorbar_fraction = 0.05
    scatter_fraction = 1 - colorbar_fraction

    # Set the width of the figure based on number of features
    single_plot_width = 0.5
    fig_height = num_features * single_plot_width

    # Now, set the total width such that fraction occupied by the scatter matrix
    # equals the height. This will let us draw a square cross-corelation matrix 
    # with the right amount of space left for the colorbar
    fig_width = fig_height / scatter_fraction

    # Set-up two subfigures, one for the scatter matrix and one for the colorbar
    fig = plt.figure(
        figsize = (fig_width, fig_height)
    )

    subfigs = fig.subfigures(
        1,
        2, 
        wspace = 0,
        hspace = 0,
        width_ratios = [scatter_fraction, colorbar_fraction]
    )

    axs1 = subfigs[0].subplots(
        num_features,
        num_features,
        gridspec_kw = {'hspace': 0, 'wspace': 0}
    )

    # Get the colormap
    cmap = mpl.colormaps['viridis']

    # Construct a normalization function to map correlation values 
    # onto the colormap
    norm = mpl.colors.Normalize(vmin = min(correlations), vmax = max(correlations))

    # Counters to keep track of where we are in the grid
    plot_count = 0
    row_count = 0
    column_count = 0

    # Loop to draw each plot
    for feature_pair, correlation, ax in zip(feature_pairs, correlations, axs1.flatten()):

        first_feature = feature_pair[0]
        second_feature = feature_pair[1]

        ax.scatter(
            features_df[first_feature],
            features_df[second_feature],
            s = 0.1,
            color = [cmap(norm(correlation))],
            alpha = 0.1
        )

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # Set y label for first plot in each row only
        if column_count == 0:
            ax.set_ylabel(first_feature)
        else:
            ax.set_ylabel('')

        # Update grid counters
        plot_count += 1
        column_count += 1

        if column_count == num_features:
            column_count = 0
            row_count += 1

        # Set x label for plots in last row only
        if row_count == num_features:
            ax.set_xlabel(first_feature)
        else:
            ax.set_xlabel('')

        ax.set_xlabel(second_feature)

    plt.tight_layout()

    axs2 = subfigs[1].subplots(1, 1)
    color_bar = mpl.colorbar.ColorbarBase(
        ax = axs2,
        cmap = cmap,
        norm = norm
    )

    color_bar.set_label('Spearman correlation coefficient', size = 18)
    color_bar.ax.tick_params(labelsize = 14) 

    return plt


def plot_cross_validation(plots: list, results: dict) -> plt:
    '''Takes a list of independent variables and the results dictionary,
    makes and returns boxplots.'''

    # Set-up the subplots
    num_conditions = len(set(results['Condition']))
    _, axes = plt.subplots(5, 1, figsize=(7, num_conditions + 3))

    # Draw each boxplot
    for plot, ax in zip(plots, axes.flatten()):
        sns.boxplot(
            y = 'Condition',
            x = plot,
            data = pd.DataFrame.from_dict(results),
            orient = 'h',
            ax = ax
        )

    plt.tight_layout()

    return plt


def plot_two_factor_cross_validation(plots: list, results: dict) -> plt:
    '''Takes a list of independent variables and the results dictionary,
    makes and returns boxplots.'''

    # Set-up the subplots
    num_conditions = len(set(results['Condition']))
    _, axes = plt.subplots(4, 1, figsize=(7, num_conditions + 1))

    # Draw each boxplot
    for plot, ax in zip(plots, axes.flatten()):
        sns.boxplot(
            y = 'Condition',
            x = plot,
            hue = 'Optimized',
            data = pd.DataFrame.from_dict(results),
            orient = 'h',
            ax = ax
        )

    plt.tight_layout()

    return plt


def make_optimization_plot(trials: dict) -> plt:
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
            sorted_bin_results['mean_test_binary_cross_entropy'] + sorted_bin_results['std_test_binary_cross_entropy'], # pylint: disable=line-too-long
            sorted_bin_results['mean_test_binary_cross_entropy'] - sorted_bin_results['std_test_binary_cross_entropy'], # pylint: disable=line-too-long
            alpha = 0.5
        )

        ax.plot(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_test_binary_cross_entropy'],
            label = 'Validation'
        )

        ax.fill_between(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_train_binary_cross_entropy'] + sorted_bin_results['std_train_binary_cross_entropy'], # pylint: disable=line-too-long
            sorted_bin_results['mean_train_binary_cross_entropy'] - sorted_bin_results['std_train_binary_cross_entropy'], # pylint: disable=line-too-long
            alpha = 0.5
        )

        ax.plot(
            sorted_bin_results['rank_test_negated_binary_cross_entropy'],
            sorted_bin_results['mean_train_binary_cross_entropy'],
            label = 'Training'
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
        features = xgb_funcs.prep_data(
            features_df = features_df,
            feature_drops = ['Fragment length (words)', 'Source', 'String'],
            shuffle_control = False
        )

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


def plot_bandwidth_scan(result: pd.DataFrame, data: np.array) -> plt:
    '''Takes results dataframe from kde_bandwidth_scan() and plots it.'''

    # Set up a 1 x 2 figure for some diagnostic plots
    fig, axs = plt.subplots(1, 2, figsize = (10, 4))

    # Add figure title
    fig.suptitle('Gaussian kernel density estimation: bandwidth selection\n', y=1.01, fontsize='x-large')

    # Add labels and the histogram to each plot
    axs[0].set_title('SciPy gaussian_kde')
    axs[0].hist(data, bins = 100, density = True, color = 'grey', label = 'Data')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('density')

    axs[1].set_title('scikit-learn KernelDensity')
    axs[1].hist(data, bins = 100, density = True, color = 'grey')
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('density')

    # Get bandwidths used in experiment
    bandwidths = result['Bandwidth'].unique()

    # Loop on bandwidths
    for bandwidth in bandwidths:

        bandwidth_data = result[result['Bandwidth'] == bandwidth]

        # Plot SciPy data
        scipy_data = bandwidth_data[bandwidth_data['Package'] == 'SciPy']
        axs[0].plot(scipy_data['x'], scipy_data['y'], label = f'Bandwidth: {bandwidth}')

        # Plot scikit-learn data
        scikit_data = bandwidth_data[bandwidth_data['Package'] == 'scikit-learn']
        axs[1].plot(scikit_data['x'], scikit_data['y'])

    fig.subplots_adjust(right = 0.78)
    fig.legend(loc = 7)
    
    return plt