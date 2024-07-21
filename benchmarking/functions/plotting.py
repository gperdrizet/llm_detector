'''Matplotlib plotting functions for data analysis'''

from __future__ import annotations
from typing import List
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_datasets(
    datasets: dict[str: pd.DataFrame] = None,
    column_renaming_dict: dict[str: str] = None,
    value_renaming_dict: dict[str: str] = None,
    data_path: str = None
) -> dict[str: pd.DataFrame]:

    '''Loads and prepares JSON or JSON lines datasets for plotting. Fixes string 
    OOM and NAN values.Translates column names and string values for pretty 
    plotting. Takes a dict of dataset name and file, returns a dict of dataset 
    name and dataframe.'''

    # Empty dict to hold results
    results = {}

    # Loop on input dicts
    for dataset_name, dataset_file in datasets.items():

        # Check if we are loading JSON or JSON lines and do the right thing
        if dataset_file.split('.')[-1] == 'json':
            dataframe = pd.read_json(f'{data_path}/{dataset_file}')

        elif dataset_file.split('.')[-1] == 'jsonl':
            dataframe = pd.read_json(
                f'{data_path}/{dataset_file}', lines = True, orient = 'records')

        # Format/translate values and column names for pretty printing in plot
        dataframe = replace_strings(
            df = dataframe,
            column_renaming_dict = column_renaming_dict,
            value_renaming_dict = value_renaming_dict
        )

        # Convert string 'OOM' and 'NAN' to np.nan
        dataframe = clean_nan_oom(dataframe)

        # Drop rows with np.nan
        # pylint: disable=E1101
        dataframe.dropna(inplace = True)
        # pylint: enable=E1101

        # Add cleaned dataframe to results
        results[dataset_name] = dataframe

    return results

def clean_nan_oom(df: pd.DataFrame=None) -> pd.DataFrame:
    '''Replaces string NAN and OOM values with np.nan'''

    df.replace('NAN', np.nan, inplace=True)
    df.replace('OOM', np.nan, inplace=True)

    return df

def replace_strings(
    df: pd.DataFrame=None,
    column_renaming_dict: dict=None,
    value_renaming_dict: dict=None
) -> pd.DataFrame:

    '''Takes pandas df with loading time data, replaces string variable
    names and values with more human readable strings.'''

    # Do the column name substitutions
    df.rename(columns=column_renaming_dict, inplace=True)

    # Do the value substitutions
    df.replace(value_renaming_dict, regex=True, inplace=True)

    return df


def exclude_factor_levels(
        data: pd.DataFrame = None,
        exclusions: dict[str: List[str]] = None
) -> pd.DataFrame:

    '''Removes a list of levels from factor in dataframe'''

    if exclusions is not None:
        for factor, levels in exclusions.items():
            for level in levels:
                data=data[data[factor] != level]

    return data


def multipanel_error_bar_two_factors(
    figure_title: str=None,
    data: pd.DataFrame=None,
    x_axis_extra_pad: float=0,
    y_axis_extra_pad: float=0,
    font_size: str='13',
    plot_rows: int=2,
    plot_cols: int=2,
    plot_width: float=3.0,
    plot_height: float=3.0,
    exclusions: dict[str, list]=None,
    panel_factor: str=None,
    series_factor: str=None,
    independent_var: str=None,
    dependent_var: str=None,
    legend_loc: str='best',
    legend_font: str='small'
) -> plt.Axes:

    '''Generalized scatter plot with error bars (mean +/- STD). Makes and 
    returns multi panel figure with a single independent and dependent variable 
    and two additional factors. Levels of the first factor are the subplot 
    panels and the levels of the second factor are the series on each plot.'''

    # Exclude any factors/levels
    data = exclude_factor_levels(
            data = data,
            exclusions = exclusions
    )

    # Get dataset wide dependent and independent variable min and max to set axis scales
    dependent_max=max(data[dependent_var])
    dependent_min=min(data[dependent_var])
    independent_max=max(data[independent_var])
    independent_min=min(data[independent_var])

    # Get unique values of factors for plots and series
    panel_factor_levels=data[panel_factor].unique()
    series_factor_levels=data[series_factor].unique()

    # Set general font size
    plt.rcParams['font.size']=font_size

    # Set-up figure and axis array
    fig, axs=plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(plot_width*plot_cols, plot_height*plot_rows),
        sharex='col',
        sharey='row',
        tight_layout=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    # Fix dimensionality of axis object for one row figures
    if len(axs.shape) == 1:
        axs=np.array([axs])

    # Construct list of tuples specifying the axs indices for each subplot
    plots=list(itertools.product(range(plot_rows), range(plot_cols)))

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

        # Calculate axis ranges
        y_min=dependent_min - (dependent_min * 0.1) - y_axis_extra_pad
        y_max=dependent_max + (dependent_max * 0.1) + y_axis_extra_pad
        x_min=independent_min - (independent_min * 0.5) - x_axis_extra_pad
        x_max=independent_max + (independent_max * 0.1) + x_axis_extra_pad

        # Set y-axis range
        axs[plot[0], plot[1]].set_ylim(y_min, y_max)

        # Set x-axis range
        axs[plot[0], plot[1]].set_xlim(x_min, x_max)

    # Set figure title
    fig.text(0.5, 1, figure_title, ha='center', fontsize='x-large')

    # Set single label for shared x
    fig.text(0.5, 0.01, independent_var, ha='center')

    # Set single label for shared y
    fig.text(0.01, 0.5, dependent_var, va='center', ha='center', rotation=90)

    # Add legend only on upper right plot
    axs[0,1].legend(
        title=series_factor,
        fontsize=legend_font,
        loc=legend_loc
    )

    return plt


def single_errorbar(
    figure_title: str=None,
    data: pd.DataFrame=None,
    plot_width: float=4.0,
    plot_height: float=4.0,
    exclusions: dict[str, list]=None,
    series_factor: str=None,
    independent_var: str=None,
    dependent_var: str=None,
    legend_loc: str=None,
    legend_font: str=None
) -> plt.Axes:

    '''Generalized single error bar plot with one additional factor for data series.'''

    # Exclude any factors/levels
    data = exclude_factor_levels(
            data = data,
            exclusions = exclusions
    )

    # Set the plot size
    plt.subplots(figsize=(plot_width, plot_height))

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
    plt.legend(
        title=series_factor,
        fontsize=legend_font,
        loc=legend_loc
    )
    
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)

    # Return plot
    return plt
