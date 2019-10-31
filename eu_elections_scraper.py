import numpy as np
import janitor

from scrape_table import *
from polling_analysis import *

def missing_vals(val):
    if val == "–":
        return None
    return val

def cleaning(table):

    df = (
        table.query("area in ['GB', 'UK']")
        .query("samplesize != '–'")
        .rename({
            "pollster_client_s_": "polling_organisation_client", 
            "date_s_conducted": "date", 
            "samplesize": "sample_size",
            "plaid_cymru": "plaid",
            "other": "others",
            "brexit_party": "brexit",
        }, axis=1)
    )

    df.loc[:, "sample_size"] = df.sample_size.str.replace("?", "2,000")

    columns = df.columns [4:]

    for col in columns:

        series = df[col].str.replace("?", "0")

        df.loc[:, col] = series.map(missing_vals)

    return df

if __name__ == "__main__":

    raw_html = get_html_content("https://en.wikipedia.org/wiki/2019_European_Parliament_election_in_the_United_Kingdom")

    processed_html = BeautifulSoup(raw_html, 'html.parser')

    all_tables = read_html(processed_html)

    polling = all_tables[8].clean_names() # obviously this could change in future

    ignore = ["plaid", "snp", "ukip"]

    all_polls = (
        cleaning(polling)
            .pipe(render_numeric)
            .pipe(parse_polling_dates)
            .pipe(parse_polling_org)
            .pipe(parse_minor_parties)
            .set_index("date")
            .drop(ignore, axis=1)
    )


    f, ax = plt.subplots()

    for pollster, style in zip(["YouGov", "Opinium", "ComRes"], ["-", "--", ":"]):

        subset = all_polls.query(f"polling_org == '{pollster}'")

        party_colours = colour_defs()

        for k in ignore:
            party_colours.pop(k) 

        parties = list(party_colours.keys())
        palette = list(party_colours.values())

        subset.plot(
            y=parties,
            ax=ax,
            color=palette,
            style=style,
            alpha=0.7,
            legend=False,
            rot=0,
            label=None,
            linewidth=2
        )

        subset.plot(
            y=parties,
            ax=ax,
            color=palette,
            style=".",
            alpha=0.7,
            legend=False,
            rot=0,
            label=None,
            linewidth=2
        )

    name_lookup = clean_party_names()

    ax.legend(
        handles = [
            mpatches.Patch(color=v, label=name_lookup[k])
            for k, v in party_colours.items()
            if k not in ignore
        ],
        fontsize="xx-large",
        ncol = 3,
    )

    ax.set_xlabel("Date", fontsize="x-large")
    ax.set_ylabel("%", fontsize="x-large")

    ax.set_ylim(0, 40)

    ax.set_xlim(pd.datetime(2019,4,1), pd.datetime(2019,5,23))

    ax.tick_params(labelsize="large", rotation=30)

    plt.tight_layout()


