'''Collection of plot functions refactored from notebooks'''

# PyPI imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def fix_labels(mylabels, tooclose=0.1, sepfactor=2):
    '''Fixes label spacing for pie charts'''

    vecs=np.zeros((len(mylabels), len(mylabels), 2))
    dists=np.zeros((len(mylabels), len(mylabels)))

    for i in range(0, len(mylabels)-1):
        for j in range(i+1, len(mylabels)):
            a=np.array(mylabels[i].get_position())
            b=np.array(mylabels[j].get_position())
            dists[i,j]=np.linalg.norm(a-b)
            vecs[i,j,:]=a-b

            if dists[i,j] < tooclose:
                mylabels[i].set_x(a[0] + sepfactor*vecs[i,j,0])
                mylabels[i].set_y(a[1] + sepfactor*vecs[i,j,1])
                mylabels[j].set_x(b[0] - sepfactor*vecs[i,j,0])
                mylabels[j].set_y(b[1] - sepfactor*vecs[i,j,1])


def data_composition(data_df: pd.DataFrame) -> plt:
    '''Plots authorship and source composition of dataset as pie charts'''

    authors=data_df['Author'].value_counts()
    datasets=data_df['Source'].value_counts()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3))
    fig.suptitle('Dataset composition')

    axs[0].set_title('Author')
    _, labels=axs[0].pie(authors, labels=authors.index)
    fix_labels(labels, sepfactor=6)

    axs[1].set_title('Data source')
    axs[1].pie(datasets, labels=datasets.index)

    plt.tight_layout()

    return plt


def length_distributions(
        title: str,
        data_df: pd.DataFrame,
        rows: int=3,
        cols: int=2,
        hue_by:
        bool=None
) -> plt:

    '''Draws text length distribution plots.'''

    # Layout math
    std_width=7.5
    height=((std_width / cols) - 0.5) * rows
    height=height*0.75

    # Setup figure
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(std_width, height))
    axs=axs.flatten()
    fig.suptitle(title)

    # Plot distribution for each dataset
    datasets=data_df['Source'].value_counts()

    for i, dataset in enumerate(datasets.index):

        # Get the data
        plot_data_df=data_df[data_df['Source'] == dataset]

        # Draw the plot
        axs[i].set_title(f'Dataset: {dataset}')

        dist=sns.kdeplot(
            data=plot_data_df,
            x='Text length (words)',
            hue=hue_by,
            log_scale=10,
            legend=True,
            ax=axs[i]
        )

        if hue_by is not None:
            dist.get_legend().set_title(None)

    # Plot the combined distribution for the last plot
    axs[-1].set_title('Dataset: combined')
    dist=sns.kdeplot(
        data=data_df,
        x='Text length (words)',
        hue=hue_by,
        log_scale=10,
        legend=True,
        ax=axs[-1]
    )

    if hue_by is not None:
        dist.get_legend().set_title(None)

    plt.tight_layout()

    return plt


def bin_density_scatter(
        title: str,
        data_df: pd.DataFrame,
        rows: int=3,
        cols: int=2,
        num_bins: int=100
) -> plt:

    '''Bins data by length, gets bin density for human and synthetic text,
    plots the two against each other as scatter, coloring points by text length.'''

    # Layout math
    std_width=7.5
    height=((std_width / cols) - 0.6) * rows

    # Setup figure
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(std_width, height))
    axs=axs.flatten()
    fig.suptitle(title)

    # Loop on the datasets from each source
    datasets=data_df['Source'].value_counts()

    for i, dataset in enumerate(datasets.index):
        plot_data_df=data_df[data_df['Source'] == dataset]

        # Get bins for combined human and synthetic data
        _, bins=np.histogram(plot_data_df['Text length (words)'], bins=num_bins)

        # Use the bin edges to get densities for human and synthetic data separately
        human_density, _=np.histogram(
            plot_data_df['Text length (words)'][plot_data_df['Synthetic'] == 'Human'],
            bins=bins,
            density=True
        )
        synthetic_density, _=np.histogram(
            plot_data_df['Text length (words)'][plot_data_df['Synthetic'] == 'Synthetic'],
            bins=bins,
            density=True
        )

        # Plot the bin densities against each other
        this_plot=axs[i].scatter(x=human_density, y=synthetic_density, c=bins[:-1])
        fig.colorbar(this_plot)
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].set_ylabel('Synthetic length bin density')
        axs[i].set_xlabel('Human length bin density')
        axs[i].set_title(f'Dataset: {dataset}')

    # Do the same for the complete dataset
    _, bins=np.histogram(data_df['Text length (words)'], bins=num_bins)
    human_density, _=np.histogram(
        data_df['Text length (words)'][data_df['Synthetic'] == 'Human'],
        bins=bins,
        density=True
    )
    synthetic_density, _=np.histogram(
        data_df['Text length (words)'][data_df['Synthetic'] == 'Synthetic'],
        bins=bins,
        density=True
    )

    # Add the combined data as the last plot
    last_plot=axs[-1].scatter(x=human_density, y=synthetic_density, c=bins[:-1])
    axs[-1].set_yscale('log')
    axs[-1].set_xscale('log')
    axs[-1].set_ylabel('Synthetic length bin density')
    axs[-1].set_xlabel('Human length bin density')
    axs[-1].set_title('Dataset: combined')
    fig.colorbar(last_plot)

    plt.tight_layout()

    return plt
