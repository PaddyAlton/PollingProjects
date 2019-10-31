import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_color_codes()

from scrape_table import download_and_transform

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

def plot_timeseries(y, data, **kwargs):

    ax = data[y].plot(style=".", **kwargs)

    return ax

if __name__ == "__main__":

    all_polls = download_and_transform()

    party_colours = colour_defs()

    for party in party_colours.keys():
        ax = plot_timeseries(party, all_polls, color=party_colours[party])
