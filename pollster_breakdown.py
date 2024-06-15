# pollster_breakdown.py

import numpy as np

import pandas as pd
import pandas_flavor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

import seaborn as sns; sns.set_style("white"); sns.set_color_codes()

import statsmodels.api as smod

from datetime import date

from scrape_table import download_and_transform
from polling_analysis import poll_plotter


def adjust_data(full_polling_data: pd.DataFrame) -> pd.DataFrame:

    polling_data = full_polling_data.loc["2023-12-01":]

    party_columns = list(polling_data.columns[-8:-1])

    relevant_columns = ["sample_size", "polling_org"] + party_columns

    long_format = (
        polling_data[relevant_columns]
        .reset_index()
        .melt(
            id_vars=["date", "sample_size", "polling_org"],
            value_vars=party_columns,
            var_name="party",
            value_name="vote_share",
        )
    )

    pollster_average = long_format.groupby(["polling_org", "party"]).vote_share.mean().rename("avg")

    party_avg = long_format.groupby("party").vote_share.mean().rename("party_avg")
    
    vote_share_data = long_format.set_index(["polling_org", "party"])

    with_avg = vote_share_data.join(pollster_average)
    with_avg["normed_vote_share"] = with_avg.vote_share.div(with_avg.avg)
    with_avg = with_avg.reset_index("polling_org").join(party_avg)
    with_avg["adjusted_vote_share"] = with_avg.party_avg.mul(with_avg.normed_vote_share)

    adjusted_data = with_avg.reset_index().set_index("date")

    adjusted_data = (
        with_avg
        .reset_index()
        .pivot_table(index=["date", "sample_size"], columns="party", values="adjusted_vote_share")
        .reset_index("sample_size")
    )

    return adjusted_data


if __name__ == "__main__":

    polling_data = download_and_transform().sort_index()

    adjusted_data = adjust_data(polling_data)

    f, ax = plt.subplots()

    ax = poll_plotter(adjusted_data, ax)

    last_election_date = date(2019,12,19)
    election_date = date(2024,7,4)

    xlim = ax.set_xlim(date(2024,1,1), election_date)

    election_called = ax.axvline(date(2024,5,22), color="k", zorder=-1)

    ax.set_title(
        "Opinion Polling for the next UK general election\n(normalised across pollsters)",
        fontsize="xx-large",
    )

    plt.grid(axis="y")

    plt.show()
