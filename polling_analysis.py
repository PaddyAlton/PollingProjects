# polling_analysis.py
# script for plotting and analysing polling data scraped from Wikipedia

import numpy as np
import pandas as pd
import pandas_flavor
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()

from scrape_table import download_and_transform

@pandas_flavor.register_dataframe_method
def get_sampling_uncertainty(df, named_column, sample_size_column):

    """
    get_sampling_uncertainty

        sigma = SQRT( p * (1-p) / N )

    We register this function as a new pandas DataFrame method

    INPUTS:
        df - self; the dataframe object
        named_column - the column on which to calculate the uncertainties
        sample_size_column - the column from which to obtain sample sizes

    OUTPUTS:
        df - the modified dataframe object (i.e. self)

    """

    sample_size = df[sample_size_column]

    using_percentages = df[named_column].max() > 1

    data_series = (df[named_column] / 100) if using_percentages else df[named_column]

    uncertainty = (
        data_series
            .mul(data_series.sub(1.0).mul(-1))
            .div(sample_size)
            .map(np.sqrt)
    )

    if not using_percentages:
        return uncertainty
    return uncertainty * 100

def get_party_list():
    return [
        "con", "lab", "lib_dem", "ukip", "green", "snp", "plaid"
    ]

def get_palette():
    return [
        "b", "r", "orange", "purple", "g", "y", "darkgreen"
    ]

def add_trendline(data, plot_clr, ax=None, window="14d"):

    """
    add_trendline

    This function converts a Series extracted from a DataFrame into
    a rolling average (default window = 14 days) over the Series. This is
    then plotted onto a supplied plotting axis (or a fresh one if not
    supplied)

    """

    trendline = (
        data.sort_index()
            .rolling(window).mean()
            .resample("d").mean()
            .interpolate("time")
    )


    if ax is None:
        f, ax = plt.subplots()

    ax.plot(trendline, linewidth=3, alpha=0.7, c=plot_clr)

    return ax

def format_plot(ax):
    """ format_plot controls plot formatting when the data is displayed """
    ax.set_ylim(0, 55)
    return ax

def plot_data(polling_data):

    f, ax = plt.subplots()
    f.set_tight_layout(True)

    parties = get_party_list()
    palette = get_palette()

    polling_data.plot(
        y=parties,
        ax=ax,
        style=".",
        color=palette,
        alpha=0.7,
        legend=False,
    )

    for party, clr in zip(parties, palette):
        ax = add_trendline(polling_data[party], clr, ax=ax)

    ax = format_plot(ax)

    return ax

if __name__ == "__main__":

    polling_data = download_and_transform()

    ax = plot_data(polling_data)
