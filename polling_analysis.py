# polling_analysis.py
# script for plotting and analysing polling data scraped from Wikipedia

import numpy as np

import pandas as pd
import pandas_flavor

import matplotlib.pyplot as plt; plt.ion()
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

import seaborn as sns; sns.set_style("white"); sns.set_color_codes()

import statsmodels.api as smod

from scrape_table import download_and_transform

if not hasattr(pd.DataFrame, "get_sampling_uncertainty"):
    # register a new method to get uncertainty from random sampling
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
                .add(0.005**2) # add in uncertainty due to rounding of figures
                .map(np.sqrt)
        )

        uncertainty.fillna(uncertainty.mean(), inplace=True)

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
        "plaid_cymru": "#008142",
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
        "plaid_cymru": "Plaid Cymru",
        "ukip": "UKIP",
        "green": "Green",
        "change_uk": "ChUK",
        "brexit": "Brexit",
    }

def get_party_list():
    return [
        "con", "lab", "lib_dem", "ukip", "green", "snp", "plaid_cymru"
    ]

def get_palette():
    return [
        "b", "r", "orange", "purple", "g", "y", "darkgreen"
    ]

def centered_ticklabels(ax):

    """
    centered_ticklabels

    Centre the tick labels properly

    """

    years = []

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")

    return ax

def add_trendline(data, uncertainty, plot_clr, ax=None):

    """
    add_trendline

    This function converts a Series extracted from a DataFrame into
    a rolling average using LOWESS (locally weighted scatterplot
    smoothing). The procedure is a nine-poll smoothing. The function
    adds the line to a plot and also adds the polling uncertainty
    to it as a shaded region.

    """

    n_vals = data.dropna().size

    smoothed_data = smod.nonparametric.lowess(data.values, data.index, frac=9/n_vals)

    trendline = pd.Series(
        smoothed_data[:, 1],
        index = pd.to_datetime(smoothed_data[:, 0]),
    )

    if ax is None:
        f, ax = plt.subplots()

    ax.plot(trendline, linewidth=3, alpha=0.7, c=plot_clr)

    error = np.interp(trendline.index, uncertainty.index, uncertainty.values)

    ax.fill_between(
        trendline.index,
        trendline-error,
        trendline+error,
        color=plot_clr,
        alpha=0.5,
    )

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
        uncertainty = polling_data.get_sampling_uncertainty(party, "samplesize").mul(2.0)
        ax = add_trendline(data, uncertainty, clr, ax=ax)

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

    ax.set_title("Opinion polling for the next UK general election", fontsize="xx-large")
    ax.set_xlabel("Date", fontsize="x-large")
    ax.set_ylabel("%", fontsize="x-large")

    ax.set_ylim(0, 55)

    # tick labelling:

    ax.tick_params(labelsize="large")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    locator = ax.xaxis.get_major_locator()

    concise_date_formatter = mdates.ConciseDateFormatter(
        locator,
        formats = ["%Y", "%B", "%d", "%H:%M", "%H:%M", "%S.%f"],
        zero_formats = ["", "%B\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
        show_offset = False,
    )

    ax.xaxis.set_major_formatter(concise_date_formatter)

    ax = centered_ticklabels(ax)

    return ax

if __name__ == "__main__":

    polling_data = download_and_transform().sort_index()

    f, ax = plt.subplots()

    ax = poll_plotter(polling_data, ax)

    ax.set_xlim(pd.datetime(2019,1,1), pd.datetime(2019,12,12))
