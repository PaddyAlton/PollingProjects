import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_color_codes()

from scrape_table import download_and_transform

def plot_timeseries(y, data, **kwargs):

    ax = data[y].plot(style=".", **kwargs)

    return ax

if __name__ == "__main__":

    all_polls = download_and_transform()

    ax = plot_timeseries("con", all_polls)
