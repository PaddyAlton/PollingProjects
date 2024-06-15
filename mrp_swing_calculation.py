# mrp_swing_calculation.py

import geopandas as gpd
from functools import partial
import pandas as pd

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


def import_full_dataset() -> gpd.GeoDataFrame:
    """
    import_full_dataset

    Returns a GeoDataFrame containing constituency boundaries and YouGov
    MRP-based projected results
        
    """
    # the .geojson with boundaries and IDs
    # from 
    # https://www.data.gov.uk/dataset/78e0c4f0-237f-41be-a81e-9888a8d93f28/westminster-parliamentary-constituencies-july-2024-boundaries-uk-bfc
    constituency_df = gpd.read_file("assets/constituencies_2024_BFC.geojson")

    # the latest YouGov projections for each constituency
    mrp_results = pd.read_excel("yougov_mrp/results_030624.xlsx")

    # join via complex condition: either the constituency code or name matches
    combined_df = (
        constituency_df
        .join(mrp_results, how="cross")
        .query("PCON24NM == constituency | PCON24CD == const")
    )

    return combined_df


def con_to_reform_swing(row: pd.Series, points: int|float) -> pd.Series:
    """
    con_to_reform_swing

    This function is designed to be applied to a projected set of
    voting intentions, modelling the effect on the result of a uniform
    1 percentage point swing from the Conservatives to Reform

    """
    swing = points / 100.0

    adjusted_row = row.copy()

    sharecols = [
        "ConShare",
        "LabShare",
        "LibDemShare",
        "GreenShare",
        "ReformShare",
        "PlaidShare",
        "SNPShare",
        "OthersShare",
    ]

    adjusted_row["ReformShare"] = row["ReformShare"] + swing
    adjusted_row["ConShare"] = row["ConShare"] - swing

    party_lookup = {
        "ConShare": "Conservatives",
        "LabShare": "Labour",
        "LibDemShare": "Lib Dems",
        "GreenShare": "Green",
        "ReformShare": "Reform",
        "PlaidShare": "Plaid",
        "SNPShare": "SNP",
        "OthersShare": "Other",
    }

    # check the winner
    winner_idx = adjusted_row[sharecols].argmax()
    winner_col = adjusted_row[sharecols].index[winner_idx]
    winner = party_lookup.get(winner_col)

    if winner:
        adjusted_row["WinnerGE2024"] = winner

    return adjusted_row


if __name__ == "__main__":

    combined_df = import_full_dataset().set_index("PCON24CD")

    party_list, colour_list = colour_map()

    swingfunc =  partial(con_to_reform_swing, points=6)

    swing_result = combined_df.apply(swingfunc, axis="columns")

    msk = swing_result.WinnerGE2024.ne(combined_df.WinnerGE2024)

    filtered_result = swing_result.loc[msk]

    mrp_map = filtered_result.explore(
        column="WinnerGE2024",
        tiles="CartoDB positron",
        cmap=colour_list,
        categories=party_list,
        style_kwds={"fillOpacity": 1.0, "color": "white", "weight": 1},
    )

    mrp_map.save("assets/mrp_map.html")
