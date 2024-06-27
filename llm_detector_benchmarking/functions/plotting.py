'''Matplotlib plotting functions for data analysis'''

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_nan_oom(df: pd.DataFrame=None) -> pd.DataFrame:
    '''Replaces string NAN and OOM values with np.nan'''

    df.replace('NAN', np.nan, inplace=True)
    df.replace('OOM', np.nan, inplace=True)

    return df

def replace_strings(
    df: pd.DataFrame=None,
    value_translation_dict: dict=None,
    column_name_translation_dict: dict=None
) -> pd.DataFrame:

    '''Takes pandas df with loading time data, replaces string variable
    names and values with more human readable strings.'''

    # Do the value substitutions
    df.replace(value_translation_dict, regex=True, inplace=True)

    # Do the column name substitutions
    df.rename(columns=column_name_translation_dict, inplace=True)

    return df

def plot_time_by_cache_device_map_cpus(
    data: pd.DataFrame
) -> plt.Axes:

    '''Makes line plot of token generation rate by device map, CPU count and cache location
    for a user specified quantization strategy.'''

    # Get dataset wide max time to set y-axis limit
    max_time=max(data['Load time (sec.)'])

    # Get unique values of independent vars
    device_maps=data['Device map'].unique()
    cache_locations=data['Cache location'].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Set number of plot rows
    plot_rows=2

    # Get number of rows and columns for figure
    plot_cols=len(device_maps) // plot_rows

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_cols,
        plot_rows,
        figsize=(6, 6),
        sharex='col',
        sharey='row',
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_cols), range(plot_rows)))

    # loop on and enumerate in the axs indices
    for i, plot in zip(range(len(plots)), plots):

        # Select data for this device map
        plot_df=data[data['Device map'] == device_maps[i]]

        # Plot a series for each cache location
        for cache_location in cache_locations:

            series_df=plot_df[plot_df['Cache location'] == cache_location]

            means=series_df.groupby(
                ['CPU cores'],
                as_index=False
            )['Load time (sec.)'].mean()

            errors=series_df.groupby(
                ['CPU cores'],
                as_index=False
            )['Load time (sec.)'].std()

            axs[plot[0], plot[1]].errorbar(
                means['CPU cores'],
                means['Load time (sec.)'],
                yerr=errors['Load time (sec.)'],
                capsize=5,
                label=cache_location,
                linestyle='dotted',
                marker='o'
            )

        # Plot title
        axs[plot[0], plot[1]].set_title(f'{device_maps[i]}', y=1.0, pad=-18)

        # Set y-axis range
        axs[plot[0], plot[1]].set_ylim(10, max_time + (max_time * 0.1))

    # Set figure title
    fig.text(0.5, 1, 'LLaMA3 loading time', ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.01, 'CPU cores', ha='center')

    # Set single label for shared y
    fig.text(0.01, 0.45, 'Time (sec.)', ha='center', rotation=90)

    # Add legend only on upper right plot
    axs[0,1].legend(
        loc=(0.3,0.4),
        title='HF cache location',
        fontsize='small'
    )

    return plt

def plot_rate_by_device_map_cpus_max_new_tokens(
    data: pd.DataFrame,
    negative_quantization: str='8-bit'
) -> plt.Axes:

    '''Makes line plot of token generation rate by device map, CPU count and max_new_tokens
    for a user specified quantization strategy.'''

    # Get dataset wide max rate to set y-axis limit
    max_rate=max(data['Generation rate (tokens per sec.)'])

    # Select 4 or 8 bit quantized rate data (+unquantized for CPU)
    quantized_rate_df=data[data['Quantization'] != negative_quantization]

    # Get unique values of independent vars
    device_maps=quantized_rate_df['Device map'].unique()
    cpu_cores=quantized_rate_df['CPU cores'].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Set number of plot rows
    plot_rows=2

    # Get number of rows and columns for figure
    plot_cols=len(device_maps) // plot_rows

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_cols,
        plot_rows,
        figsize=(6, 6),
        sharex='col',
        sharey='row',
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_cols), range(plot_rows)))

    # loop on and enumerate in the axs indices
    for i, plot in zip(range(len(plots)), plots):

        # Select data for this device map
        plot_df=quantized_rate_df[quantized_rate_df['Device map'] == device_maps[i]]

        # Plot a series for each cpu core count
        for core_count in cpu_cores:

            series_df=plot_df[plot_df['CPU cores'] == core_count]

            means=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].mean()

            errors=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].std()

            axs[plot[0], plot[1]].errorbar(
                means['Max new tokens'],
                means['Generation rate (tokens per sec.)'],
                yerr=errors['Generation rate (tokens per sec.)'],
                capsize=5,
                label=core_count,
                linestyle='dotted',
                marker='o'
            )

        # Plot title
        axs[plot[0], plot[1]].set_title(f'{device_maps[i]}', y=1.0, pad=-18)

        # Set y-axis range
        axs[plot[0], plot[1]].set_ylim(5, max_rate + (max_rate * 0.1))

    # Set figure title
    fig.text(0.5, 1, 'LLaMA3 generation rate', ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.01, 'Max new tokens', ha='center')

    # Set single label for shared y
    fig.text(0.01, 0.35, 'Rate (tokens per sec.)', ha='center', rotation=90)

    # Add legend only on upper right plot
    axs[0,1].legend(
        loc=(0.55,0.35),
        title='CPU cores',
        fontsize='small'
    )

    return plt

def plot_rate_by_quantization(rate_df: pd.DataFrame) -> plt.Axes:
    '''Makes line plot of token generation rate for 4 and 8 bit quantization
    by device map as a function of max_new_tokens'''

    # Get dataset wide max rate to set y-axis limit
    max_rate=max(rate_df['Generation rate (tokens per sec.)'])

    # Select 4 or 8 bit quantized rate data (+unquantized for CPU)
    quantized_rate_df=rate_df[rate_df['Quantization'] != 'None']

    # Get unique values of independent vars
    device_maps=quantized_rate_df['Device map'].unique()
    quantizations=quantized_rate_df['Quantization'].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Set number of plot rows
    plot_rows=1

    # Get number of rows and columns for figure
    plot_cols=len(quantizations) // plot_rows

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(3.5 * plot_cols, 3.5 * plot_rows),
        sharex=True,
        sharey=True,
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_cols), range(plot_rows)))

    # loop on and enumerate in the axs indices
    for i, plot in zip(range(len(plots)), plots):

        # Select data for this quantization strategy
        plot_df=quantized_rate_df[quantized_rate_df['Quantization'] == quantizations[i]]

        # Plot a series for each device map
        for device_map in device_maps:

            series_df=plot_df[plot_df['Device map'] == device_map]

            means=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].mean()

            errors=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].std()

            axs[plot[0]].errorbar(
                means['Max new tokens'],
                means['Generation rate (tokens per sec.)'],
                yerr=errors['Generation rate (tokens per sec.)'],
                capsize=5,
                label=device_map,
                linestyle='dotted',
                marker='o'
            )

            # Plot title
            axs[plot[0]].set_title(f'Quantization: {quantizations[i]}', y=1.0, pad=-18)

            # Set y-axis range
            axs[plot[0]].set_ylim(0, max_rate + (max_rate * 0.1))

    # Set figure title
    fig.text(0.5, 1, 'LLaMA3 generation rate', ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.02, 'Max new tokens', ha='center')

    # Only add y axis label on first plot
    axs[0].set_ylabel('Rate (tokens per sec.)')

    # Add legend only on last plot
    axs[1].legend(
        loc=(0.2,0.5),
        #title='Device map',
        fontsize='small'
    )

    return plt


def plot_rate_by_decoding_strategy(rate_df: pd.DataFrame) -> plt.Axes:
    '''Makes line plot of token generation rate for each decoding strategy
    as a function of max_new_tokens'''

    # Get dataset wide max rate to set y-axis limit
    max_rate=max(rate_df['Generation rate (tokens per sec.)'])

    # Get unique values of independent vars
    device_maps=rate_df['Device map'].unique()
    decoding_strategies=rate_df['Decoding strategy'].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Set number of plot rows
    plot_rows=1

    # Get number of rows and columns for figure
    plot_cols=len(device_maps) // plot_rows

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(3.5 * plot_cols, 3.5 * plot_rows),
        sharex=True,
        sharey=True,
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_cols), range(plot_rows)))

    # loop on and enumerate in the axs indices
    for i, plot in zip(range(len(plots)), plots):

        # Select data for this device map
        plot_df=rate_df[rate_df['Device map'] == device_maps[i]]

        # Plot a series for each decoding strategy
        for decoding_strategy in decoding_strategies:

            series_df=plot_df[plot_df['Decoding strategy'] == decoding_strategy]

            means=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].mean()

            errors=series_df.groupby(
                ['Max new tokens'],
                as_index=False
            )['Generation rate (tokens per sec.)'].std()

            axs[plot[0]].errorbar(
                means['Max new tokens'],
                means['Generation rate (tokens per sec.)'],
                yerr=errors['Generation rate (tokens per sec.)'],
                capsize=5,
                label=decoding_strategy,
                linestyle='dotted',
                marker='o'
            )

            # Plot title
            axs[plot[0]].set_title(f'{device_maps[i]}', y=1.0, pad=-18)

            # Set y-axis range
            axs[plot[0]].set_ylim(0, max_rate + (max_rate * 0.1))

    # Set figure title
    fig.text(0.5, 1, 'LLaMA3 generation rate', ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.02, 'Max new tokens', ha='center')

    # Only add y axis label on first plot
    axs[0].set_ylabel('Rate (tokens per sec.)')

    # Add legend only on last plot
    axs[1].legend(
        loc=(0.3,0.3),
        #title='Device map',
        fontsize='x-small'
    )

    return plt

def plot_memory_by_device_map_input_length(
    memory_df: pd.DataFrame
) -> plt.Axes:

    '''Plots logit calculation memory footprint by device map
    and input sequence length'''

    # Get dataset wide max memory to set y-axis limit
    max_memory=max(memory_df['Peak memory (GB)'])

    # Get unique values of independent vars
    device_maps=memory_df['Device map'].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Loop on device maps
    for device_map in device_maps:

        # Select data for this device map
        series_df=memory_df[memory_df['Device map'] == device_map]

        means=series_df.groupby(
            ['Input length (words)'],
            as_index=False
        )['Peak memory (GB)'].mean()

        errors=series_df.groupby(
            ['Input length (words)'],
            as_index=False
        )['Peak memory (GB)'].std()

        plt.errorbar(
            means['Input length (words)'],
            means['Peak memory (GB)'],
            yerr=errors['Peak memory (GB)'],
            capsize=5,
            label=device_map,
            linestyle='dotted',
            marker='o'
        )

    # Plot title
    plt.title('Logit calculation memory footprint')

    # Set y-axis range
    plt.ylim(0, max_memory + (max_memory * 0.1))

    plt.xlabel('Input length (words)')
    plt.ylabel('Peak memory (GB)')

    # Add legend only on last plot
    plt.legend(
        fontsize='x-small'
    )

    return plt

def two_by_two_error_bar_two_factors(
    figure_title: str=None,
    data: pd.DataFrame=None,
    exclusions: dict=None,
    panel_factor: str=None,
    series_factor: str=None,
    independent_var: str=None,
    dependent_var: str=None
) -> plt.Axes:

    '''Generalized error bar plot to make 2 by 2 panel figure with two
    factors. Levels of the first factor are the panels and levels of the
    second are the series on each plot.'''

    # Exclude any factors/levels
    if exclusions is not None:
        for factor, levels in exclusions.items():
            for level in levels:
                data=data[data[factor] != level]

    # Get dataset wide dependent and independent variable min and max to set axis scales
    dependent_max=max(data[dependent_var])
    dependent_min=min(data[dependent_var])
    independent_max=max(data[independent_var])
    independent_min=min(data[independent_var])

    # Get unique values of factors for plots and series
    panel_factor_levels=data[panel_factor].unique()
    series_factor_levels=data[series_factor].unique()

    # Set general font size
    plt.rcParams['font.size']='13'

    # Set number of cols and rows
    plot_rows, plot_cols=2, 2

    # Set single plot dimension
    plot_width, plot_height=3.5, 3.5

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_cols,
        plot_rows,
        figsize=(plot_width*plot_cols, plot_height*plot_rows),
        sharex='col',
        sharey='row',
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_cols), range(plot_rows)))

    # loop on and enumerate in the axs indices
    for i, plot in zip(range(len(plots)), plots):

        # Select data for this panel
        plot_df=data[data[panel_factor] == panel_factor_levels[i]]

        # Plot each series on this panel
        for series_factor_level in series_factor_levels:

            series_df=plot_df[plot_df[series_factor] == series_factor_level]

            means=series_df.groupby(
                [independent_var],
                as_index=False
            )[dependent_var].mean()

            errors=series_df.groupby(
                [independent_var],
                as_index=False
            )[dependent_var].std()

            axs[plot[0], plot[1]].errorbar(
                means[independent_var],
                means[dependent_var],
                yerr=errors[dependent_var],
                capsize=5,
                label=series_factor_level,
                linestyle='dotted',
                marker='o'
            )

        # Panel title
        axs[plot[0], plot[1]].set_title(f'{panel_factor_levels[i]}', y=1.0, pad=-18)

        # Set y-axis range
        axs[plot[0], plot[1]].set_ylim(
            dependent_min - (dependent_min * 0.2),
            dependent_max + (dependent_max * 0.1)
        )

        # Set y-axis range
        axs[plot[0], plot[1]].set_xlim(
            independent_min - (independent_min * 0.5),
            independent_max + (independent_max * 0.1)
        )

    # Set figure title
    fig.text(0.5, 1, figure_title, ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.01, independent_var, ha='center')

    # Set single label for shared y
    fig.text(0.01, 0.35, dependent_var, ha='center', rotation=90)

    # Add legend only on upper right plot
    axs[0,1].legend(
        title=series_factor,
        fontsize='small',
        loc='center right'
    )

    return plt

def single_errorbar(
    figure_title: str=None,
    data: pd.DataFrame=None,
    series_factor: str=None,
    independent_var: str=None,
    dependent_var: str=None
) -> plt.Axes:

    '''Generalized single error bar plot with one additional factor for data series.'''

    # Get the factor levels
    factor_levels=data[series_factor].unique()

    # Loop on factor levels to draw data series
    for factor_level in factor_levels:

        # Extract data for this factor level
        series_df=data[data[series_factor] == factor_level]

        # Get replicate means
        means=series_df.groupby(
            [independent_var],
            as_index=False
        )[dependent_var].mean()

        # Get replicate errors
        errors=series_df.groupby(
            [independent_var],
            as_index=False
        )[dependent_var].std()

        # Plot the series
        plt.errorbar(
            means[independent_var],
            means[dependent_var],
            yerr=errors[dependent_var],
            capsize=5,
            label=factor_level,
            linestyle='dotted',
            marker='o'
        )

    # Annotate plot
    plt.title(figure_title)
    plt.legend(loc='best')
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)

    # Return plot
    return plt