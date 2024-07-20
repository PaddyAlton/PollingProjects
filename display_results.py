from sys import argv

import numpy as np
import pandas as pd
import seaborn as sns

from branca import colormap
from geopandas import read_file


def colour_lookup() -> dict[str, str]:
    return {
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


def colour_map() -> tuple[list[str], list[str]]:
    """
    colour_map

    Custom colour map; returns two lists, one containing
    parties and the other containing hex codes

    """
    mappings = colour_lookup()

    return list(mappings.keys()), list(mappings.values())


def get_sequential_colours(hex: str) -> colormap.LinearColormap:
    H, S, L = sns.external.husl.hex_to_husl(hex)

    sats = np.linspace(0, 100, 4)

    hexcode_generator = map(lambda sat: sns.external.husl.husl_to_hex(H, sat, L), sats)

    hexcodes = [hexcode for hexcode in hexcode_generator]

    divisions = [0, 15, 30, 45]

    return colormap.LinearColormap(
        colors = hexcodes,
        vmin = 0,
        vmax = divisions[-1],
        index = divisions,
        tick_labels = divisions,
    )


if __name__ == "__main__":

    try:
        party = argv[1]
    except IndexError:
        party = "Winner"

    results = pd.read_excel("assets/ElectionMapsUK_GE2024_Supersheet.xlsx", header=0).iloc[3:, 1:38]

    real_columns_mask = [not c for c in results.columns.str.startswith("Unnamed")]

    results = results.loc[:, real_columns_mask].set_index("Code")

    gdf = read_file("assets/constituencies_2024_BGC.geojson")

    enriched_gdf = gdf.join(results, on="PCON24CD")
    enriched_gdf["Winner"] = enriched_gdf.Winner.fillna("Unknown")

    # for plotting party support
    party_gdf = (
        gdf
        .join(
            results[party]
            .div(results["TOTAL VOTES"]
            .rename(party))
            .mul(100),
            on="PCON24CD"
        )
        .join(
            results.Winner.eq(party).rename("Winner"),
            on="PCON24CD"
        )
    )

    party_gdf[party] = party_gdf[party].fillna(0)

    party_list, colour_list = colour_map()

    colorscale = get_sequential_colours(colour_lookup()[party])

    results_map = party_gdf.explore(
        column=party,
        tiles="CartoDB positron",
        #cmap=colour_list,
        cmap=colorscale,
        #categories=party_list,
        style_kwds={"fillOpacity": 1.0, "color": "grey", "weight": 1},
    )

    party_gdf.query("Winner == True").explore(
        m=results_map,
        column=party,
        tiles="CartoDB positron",
        cmap=colorscale,
        style_kwds={"fillOpacity": 1.0, "color": "blue", "weight": 2},
    )

    results_map.save(f"assets/results_map_{party}.html")
