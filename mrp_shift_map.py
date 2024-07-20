# mrp_shift_map.py

import geopandas as gpd
import pandas as pd

from mrp_swing_calculation import colour_map


def colour_map() -> tuple[list[str], list[str]]:
    """
    colour_map

    Custom colour map; returns two lists, one containing
    parties and the other containing hex codes

    """
    mappings = {
        "Conservatives": "#0087DC",
        "Labour": "#DC241F",
        "Lib Dems": "#FAA61A",
        "SNP": "#FEF987",
        "Green": "#6AB023",
        "Reform": "#12B6CF",
        "Plaid": "#008142",
    }

    return list(mappings.keys()), list(mappings.values())


def import_constituencies():
    # the .geojson with boundaries and IDs
    # from 
    # https://www.data.gov.uk/dataset/78e0c4f0-237f-41be-a81e-9888a8d93f28/westminster-parliamentary-constituencies-july-2024-boundaries-uk-bfc
    return gpd.read_file("assets/constituencies_2024_BFC.geojson")


def import_full_dataset(target_file: str) -> gpd.GeoDataFrame:
    """
    import_full_dataset

    Returns a GeoDataFrame containing constituency boundaries and YouGov
    MRP-based projected results
        
    """
    # the latest YouGov projections for each constituency
    mrp_results = pd.read_excel(f"yougov_mrp/{target_file}.xlsx")

    # join via complex condition: either the constituency code or name matches
    combined_df = (
        constituency_df
        .join(mrp_results, how="cross")
        .query("PCON24NM == area | PCON24CD == const")
    )

    return combined_df


if __name__ == "__main__":

    party_list, colour_list = colour_map()

    constituency_df = import_constituencies()

    mrp_results1 = pd.read_excel(f"yougov_mrp/results_030624.xlsx")
    mrp_results2 = pd.read_excel(f"yougov_mrp/results_190624.xlsx")

    combined1 = (
        constituency_df
        .join(mrp_results1, how="cross")
        .query("PCON24NM == constituency | PCON24CD == const")
        .set_index("PCON24CD")
        .rename({"WinnerGE2024": "early_june_winner"}, axis="columns")
    )

    combined2 = (
        constituency_df
        .join(mrp_results2, how="cross")
        .query("PCON24NM == area | PCON24CD == const")
        .set_index("PCON24CD")
        .rename({"WinnerGE2024": "late_june_winner"}, axis="columns")
    )

    shift_results = (
        combined1
        .join(combined2, lsuffix="_old")[["area", "geometry", "early_june_winner", "late_june_winner"]]
        .query("early_june_winner != late_june_winner")
    )

    mrp_map = shift_results.explore(
        column="late_june_winner",
        tiles="CartoDB positron",
        cmap=colour_list,
        categories=party_list,
        style_kwds={"fillOpacity": 1.0, "color": "white", "weight": 1},
    )

    mrp_map.save("assets/mrp_map_shifts.html")
