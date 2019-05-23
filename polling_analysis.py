# polling_analysis.py
# script for plotting and analysing polling data scraped from Wikipedia

import numpy as np
import pandas as pd
import pandas_flavor
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.patches as mpatches
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

def colour_defs():
    """
    colour_defs

    Returns a lookup dictionary, hex-codes for each party
    """
    return {
        "con": "#0087DC",
        "lab": "#DC241F",
        "lib_dem": "#FAA61A",
        "snp": "#FEF987",
        "plaid": "#008142",
        "ukip": "#70147A",
        "green": "#6AB023",
        "change_uk": "#222221",
        "brexit": "#12B6CF",
    }

def clean_party_names():
    """
    clean_party_names

    Returns a dictionary lookup from shorthand names to display names
    for parties
    """
    return {
        "con": "Con",
        "lab": "Lab",
        "lib_dem": "Lib Dem",
        "snp": "SNP",
        "plaid": "Plaid Cymru",
        "ukip": "UKIP",
        "green": "Green",
        "change_uk": "ChUK",
        "brexit": "Brexit",
    }

def get_party_list():
    return [
        "con", "lab", "lib_dem", "ukip", "green", "snp", "plaid"
    ]

def get_palette():
    return [
        "b", "r", "orange", "purple", "g", "y", "darkgreen"
    ]

def add_trendline(data, uncertainty, plot_clr, party, ax=None, window="14d"):

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

    error = uncertainty.resample("d").mean().fillna(method="pad")

#    ax.fill_between(
#        trendline.index,
#        trendline-error,
#        trendline+error,
#        color=plot_clr,
#        alpha=0.5,
#    )

    return ax

def poll_plotter(polling_data, ax):

    party_colours = colour_defs()

    parties = list(party_colours.keys())
    palette = list(party_colours.values())

    polling_data.plot(
        y=parties,
        ax=ax,
        style=".",
        color=palette,
        alpha=0.7,
        legend=False,
        rot=0,
        label=None,
    )

    for party, clr in zip(parties, palette):
        data = polling_data[party]
        uncertainty = polling_data.get_sampling_uncertainty(party, "sample_size").mul(2.0)
        ax = add_trendline(data, uncertainty, clr, party, ax=ax)

    # format the plot
    name_lookup = clean_party_names()

    ax.legend(
        handles = [
            mpatches.Patch(color=v, label=name_lookup[k])
            for k, v in party_colours.items()
        ],
        fontsize="xx-large",
        ncol = 3,
    )

    ax.set_xlabel("Date", fontsize="x-large")
    ax.set_ylabel("%", fontsize="x-large")

    ax.set_ylim(0, 55)

    ax.tick_params(labelsize="large")

    return ax

def plot_data(polling_data):

    f, ax = plt.subplots()
    #f.set_tight_layout(True)
    poll_plotter(polling_data, ax)
    return ax

if __name__ == "__main__":

    polling_data = download_and_transform()

    ax = plot_data(polling_data)

    ax.set_xlim(pd.datetime(2019,1,1), pd.datetime(2019,5,23))
