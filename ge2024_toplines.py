# ge2024_toplines.py
# this script is intended to create some graphs of the headline results
# from the July 2024 General Election

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import poli_sci_kit
import seaborn as sns

from scrape_table import download_and_transform


party_colours_short = {
    "con": "#0087DC",
    "lab": "#DC241F",
    "lib_dems": "#FAA61A",
    "snp": "#FEF987",
    "green": "#6AB023",
    "reform": "#12B6CF",     
    "others": "#D3D3D3",
}

party_colours = {
    "CON": "#0087DC",
    "LAB": "#DC241F",
    "LDM": "#FAA61A",
    "SNP": "#FEF987",
    "GRN": "#6AB023",
    "RFM": "#12B6CF",
    "PLC": "#008142",
    "IND": "#FC0FC0",
    "SPKR": "#000000",
    "DUP": "#D46A4C",
    "SF": "#326760",
    "SDLP": "#2AA82C",
    "ALL": "#F6CB2F",
    "TUV": "#0C3A6A",
    "UUP": "#48A5EE",        
    "Unknown": "#D3D3D3",
}

party_names_full = {
    "con": "Conservatives",
    "lab": "Labour",
    "lib_dems": "Liberal Democrats",
    "snp": "SNP",
    "green": "Green",
    "reform": "Reform",     
    "others": "Others",
    "CON": "Conservatives",
    "LAB": "Labour",
    "LDM": "Liberal Democrats",
    "SNP": "Scottish National Party",
    "GRN": "Green",
    "RFM": "Reform",
    "PLC": "Plaid Cymru",
    "IND": "Independent",
    "DUP": "Democratic Unionist Party",
    "SF": "Sinn Féin",
    "SDLP": "Social Democratic and Labour Party",
    "ALL": "Alliance Party",
    "TUV": "Traditional Unionist Voice",
    "UUP": "Ulster Unionist Party",   
}


def get_headlines(results: pd.DataFrame) -> pd.DataFrame:

    numeric_columns = results.columns[6:14].to_list() + ["TOTAL VOTES"]

    numeric_data = results.set_index(["Code", "Constituency + 2019 Notional Result"])[numeric_columns].fillna(0)

    headlines = numeric_data.sum().div(numeric_data["TOTAL VOTES"].sum()).mul(100).iloc[:-1]

    headlines.index.name = "Party"

    return headlines.to_frame("Vote Share (%)")


def get_seat_counts(results: pd.DataFrame) -> pd.DataFrame:
    seat_counts = results.groupby("Winner").Code.count().sort_values(ascending=False)

    seat_counts.index.name = "Party"

    return seat_counts.to_frame("Seats Won")


def get_polling_data() -> tuple[pd.DataFrame, pd.Series]:

    polling_data = download_and_transform().sort_index()

    final_polls = polling_data.loc["2024-07-02":"2024-07-03"]

    numeric_cols = polling_data.columns[5:-1]

    ge_results = polling_data.loc["2024-07-04"].set_index("area").loc[:, numeric_cols]

    uk_factor = ge_results.loc["GB"] / ge_results.loc["UK"]

    uk_poll_mask = final_polls.area.eq("GB")

    final_polls.loc[uk_poll_mask, numeric_cols] = final_polls.loc[uk_poll_mask, numeric_cols].mul(uk_factor)

    melt_polls = final_polls.melt(
        id_vars="polling_org",
        value_vars=numeric_cols,
        var_name="Party",
        value_name="Vote Share (%)",
    )

    return melt_polls, ge_results.loc["GB"]


def plot_seats():

    results = pd.read_excel("assets/ElectionMapsUK_GE2024_Supersheet.xlsx", header=0).iloc[3:, 1:38]

    gb_results = results.query("Region != 'Northern Ireland'")
    ni_results = results.query("Region == 'Northern Ireland'")

    gb_numeric_columns = gb_results.columns[6:14].to_list() + ["TOTAL VOTES"]

    gb_numeric = gb_results.set_index(["Code", "Constituency + 2019 Notional Result"])[gb_numeric_columns].fillna(0)
    full_numeric = results.set_index(["Code", "Constituency + 2019 Notional Result"])[gb_numeric_columns].fillna(0)

    gb_party_headlines = get_headlines(gb_results)
    full_party_headlines = get_headlines(results)

    full_seats = get_seat_counts(results)

    # fig, ax = plt.subplots()

    #sns.barplot(data=full_party_headlines, x="Vote Share (%)", y="Party", hue="Party", palette=party_colours, ax=axarr[0])

    # sns.barplot(data=full_seats, y="Seats Won", x="Party", hue="Party", palette=party_colours, ax=ax)

    # ax.axhline(326, color="k", linestyle=":")

    # plt.show()

    seat_alloc_lookup = full_seats.query("Party != 'SPKR'")["Seats Won"].to_dict()

    parties = list(seat_alloc_lookup.keys())
    seat_allocations = list(seat_alloc_lookup.values())
    p_clr = [party_colours[p] for p in parties]

    fig, ax = plt.subplots()

    ax = poli_sci_kit.plot.parliament(
        allocations=seat_allocations,
        labels=parties,
        colors=p_clr,
        style="rectangle",
        num_rows=16,
        marker_size=100,
        speaker=False,
        axis=ax,
    )

    artists = ax.collections

    p_names = [party_names_full[p] for p in parties]

    ax.legend(handles=artists, labels=p_names, ncol=3, loc="center", bbox_to_anchor=[0.5, 0], fontsize="large")

    plt.show()


def plot_polling_miss():

    final_polls, ge_result = get_polling_data()

    plot_df = final_polls.query("Party != 'others'")

    fig, ax = plt.subplots()

    ax = sns.kdeplot(
        data=plot_df,
        x="Vote Share (%)",
        hue="Party",
        palette=party_colours_short,
        ax=ax,
        linewidth=2,
    )

    ax.get_legend().set_ncols(3)

    # iterate through the parties, from lowest share to highest:
    for party, result in sorted(ge_result[:-1].items(), key=lambda r: r[1]):

        clr = party_colours_short[party]

        ax.axvline(result, color=clr, linewidth=2, linestyle=":", label=party_names_full[party])

    ax.legend(ncols=6, fontsize="x-large")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 0.08)

    ax.set_xlabel("Vote Share (%)", fontsize="x-large")
    ax.set_ylabel(None)
    ax.tick_params(left=None, labelleft=None, labelsize="large")
    ax.set_title("Consensus Vote Share Predictions", fontsize="x-large")

    plt.show()


def plot_swing():

    important_columns = ["Unnamed: 0", "Constituency + 2019 Notional Result", "Code", "Turnout", "Winner", "LAB.1", "LAB.2", "LAB.3"]

    res_and_swing = (
        pd.read_excel("assets/ElectionMapsUK_GE2024_Supersheet.xlsx", header=0)
          .iloc[3:]
          .loc[:, important_columns]
    )
    
    res_and_swing.columns = [
        "Winner 2019",
        "Name",
        "Code",
        "Turnout",
        "Winner 2024",
        "Result 2024",
        "Result 2019",
        "Swing",
    ]

    res_and_swing.loc[:, "Result 2019"] = res_and_swing["Result 2019"].mul(100)
    res_and_swing.loc[:, "Result 2024"] = res_and_swing["Result 2024"].mul(100)
    res_and_swing.loc[:, "Turnout"] = res_and_swing["Turnout"].mul(100)

    fig, axes = plt.subplots(ncols=3)

    sns.scatterplot(data=res_and_swing, x="Result 2019", y="Result 2024", hue="Winner 2019", palette=party_colours, ax=axes[0])
    sns.scatterplot(data=res_and_swing, x="Result 2019", y="Result 2024", hue="Winner 2024", palette=party_colours, ax=axes[1])
    sns.scatterplot(data=res_and_swing, x="Result 2024", y="Turnout", hue="Winner 2024", palette=party_colours, ax=axes[2])

    xs = np.linspace(0, 100, 101)

    for ax in axes:
        ax.plot(xs, xs, 'k-', zorder=-1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    axes[0].set_title("2019 -> 2024 Labour Vote Shares (by 2019 Winner)", fontsize="x-large")
    axes[1].set_title("2019 -> 2024 Labour Vote Shares (by 2024 Winner)", fontsize="x-large")

    plt.show()


if __name__ == "__main__":

    plot_swing()
