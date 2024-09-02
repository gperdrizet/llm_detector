'''Collection of functions for plotting in notebooks'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def data_exploration_plot(data):
    '''Takes instance of class FeatureEngineering, makes some correlation
    and distribution diagnostic plots'''

    # Set up a 2 x 2 figure for some diagnostic plots
    fig, axs = plt.subplots(
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