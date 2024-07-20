import json
import pandas as pd
import plotly.express as px
import seaborn as sns

from geopandas import read_file
from numpy import linspace


def renamer(col):
    if col.endswith("Share"):
        return col.split("Share")[0]
    return col


gdf = read_file("assets/constituencies_2024_BFC.geojson")
mrp_results = pd.read_excel("yougov_mrp/results_190624.xlsx").rename(renamer, axis="columns")

with open("assets/constituencies_2024_BFC.geojson") as fob:
    geojson = json.load(fob)

parties = ['Con', 'Lab', 'LibDem', 'Green', 'Reform', 'Plaid', 'SNP', 'Others']

#enriched_gdf = gdf.join(mrp_results.set_index("const")[parties], on="PCON24CD")
enriched_gdf = (
    gdf
    .join(mrp_results[["area", "const"] + parties], how="cross")
    .query("PCON24NM == area | PCON24CD == const")
)

colour_defs = {
    "Con": "#0087DC",
    "Lab": "#DC241F",
    "LibDem": "#FAA61A",
    "SNP": "#FEF987",
    "Green": "#6AB023",
    "Reform": "#12B6CF",
    "Plaid": "#008142",
    "Others": "#fc0fc0",
}

def get_sequential_colours(hex: str) -> list[str]:
    H, S, L = sns.external.husl.hex_to_husl(hex)

    sats = linspace(0, 100, 5)

    hexcode_generator = map(lambda sat: sns.external.husl.husl_to_hex(H, sat, L), sats)

    hexcodes = [hexcode for hexcode in hexcode_generator]

    result = [
        [0, hexcodes[0]],
        [0.4, hexcodes[1]],
        [0.5, hexcodes[2]],
        [0.6, hexcodes[3]],
        [1, hexcodes[4]],
    ]

    return result


if __name__ == "__main__":

    party = "Reform"

    party_hex = colour_defs[party]

    colorscale = get_sequential_colours(party_hex)

    fig = px.choropleth_mapbox(
        enriched_gdf,
        locations="PCON24NM",
        geojson=geojson,
        featureidkey="properties.PCON24NM",
        color=party,
        color_continuous_scale=colorscale,
        zoom=4,
        center = {"lat": 55, "lon": 0},
        mapbox_style="carto-positron",
        hover_name="PCON24NM",
    )

    fig.show()
