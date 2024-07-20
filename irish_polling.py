# irish_polling.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smod


irish_party_colours = {
    "Aon": "#44532A", # Aontú
    "FF": "#66BB66",  # Fianna Fáil
    "FG": "#6699FF",  # Fine Gael
    "GP": "#22AC6F",  # Green Party
    "SD": "#752F8B",  # Social Democrats
    "SF": "#326760",  # Sinn Féin
    "Lab": "#CC0000", # Labour Party
    "PBP–S": "#E91D50", # People Before Profit–Solidarity
}

irish_party_full_names = {
    "Aon": "Aontú",
    "FF": "Fianna Fáil",
    "FG": "Fine Gael",
    "GP": "Green Party",
    "SD": "Social Democrats",
    "SF": "Sinn Féin",
    "Lab": "Labour Party",
    "PBP–S": "People Before Profit–Solidarity",
}

full_names_to_colour = {
    irish_party_full_names[key]: value for key, value in irish_party_colours.items()
}


if __name__ == "__main__":

    url = "https://en.wikipedia.org/wiki/Next_Irish_general_election#Opinion_polls"

    irish_polls = pd.read_html(url)[6].droplevel(1, axis="columns")

    irish_polls.loc[:, "Last date of polling"] = pd.to_datetime(irish_polls["Last date of polling"])

    irish_polls_melted = irish_polls.melt(
        id_vars="Last date of polling",
        value_vars=["SF", "FF", "FG"],# "GP", "Lab", "SD", "PBP–S", "Aon"],
        var_name="party",
        value_name="share",
    )

    irish_polls_melted.loc[:, "party"] = irish_polls_melted.party.map(irish_party_full_names)

    irish_polls_melted["share"] = irish_polls_melted.share.map(lambda x: 0 if "[" in str(x) else x).astype(float)

    ax = sns.scatterplot(
        data=irish_polls_melted,
        x="Last date of polling",
        y="share",
        hue="party",
        palette=full_names_to_colour,
    )

    for party in irish_polls_melted.party.unique():

        ser = (
            irish_polls_melted
            .query(f"party == '{party}'")
            .set_index("Last date of polling")
            .sort_index()
            .share
        )

        frac = 28/ser.size

        smooth = smod.nonparametric.lowess(ser.values, ser.index, frac=frac)

        trendline = pd.Series(smooth[:, 1], index = pd.to_datetime(smooth[:, 0]))

        trendline.plot(ax=ax, linewidth=3, color=full_names_to_colour[party], label=None)

    ax.set_ylim(0, 40)

    ax.legend(fontsize="x-large", ncols=3)

    ax.set_xlabel("Date", fontsize="large")
    ax.set_ylabel("Vote Share", fontsize="large")
    ax.set_title("Republic of Ireland Opinion Polling", fontsize="x-large")

    plt.show()
