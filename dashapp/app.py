import json
import pandas as pd
import plotly.express as px
import seaborn as sns

from dash import Dash, dcc, Input, Output, html
from geopandas import read_file
from numpy import linspace


def renamer(col):
    if col.endswith("Share"):
        return col.split("Share")[0]
    return col


gdf = read_file("assets/constituencies_2024_BFC.geojson")
mrp_results = pd.read_excel("yougov_mrp/results_030624.xlsx").rename(renamer, axis="columns")

with open("assets/constituencies_2024_BFC.geojson") as fob:
    geojson = json.load(fob)


parties = ['Con', 'Lab', 'LibDem', 'Green', 'Reform', 'Plaid', 'SNP', 'Others']

enriched_gdf = gdf.join(mrp_results.set_index("const")[parties], on="PCON24CD")

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


# app layout
app = Dash(__name__)

app.layout = html.Div(children=[
    # The interactive plotly map
    html.Div(className="row", children=[

        html.Div(className="six columns", children=[
        # The Dropdown to select the dataframes
            dcc.Dropdown(
                options=parties,
                value='Lab',
                id="party_dropdown",
                style={"width": "50%", "display": "inline-block"})
        ]),
    ]),

    html.Br(),

    html.Div(id="party_dropdown_output"),

    html.Br(),

    # The interactive plotly map
    dcc.Graph(id="mrp_map", clickData={"points": [{"customdata": "Aldershot"}]}),

    # The Line graph
    dcc.Graph(id="line_graph")
])


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


@app.callback(
    Output("party_dropdown_output", "children"),
    Input("party_dropdown", "value"),
)
def dropdown_output(value):
    return f"You have chosen the {value} party."


# Draw a plotly map based on the dropdown value chosen
@app.callback(
    Output("mrp_map", "figure"),
    Input("party_dropdown", "value"),
)
def choropleth_map(party_dropdown):

    party_hex = colour_defs[party_dropdown]

    colorscale = get_sequential_colours(party_hex)

    fig = px.choropleth_mapbox(
        enriched_gdf,
        locations="PCON24NM",
        geojson=geojson,
        featureidkey="properties.PCON24NM",
        color=party_dropdown,
        color_continuous_scale=colorscale,
        zoom=4,
        center = {"lat": 55, "lon": 0},
        mapbox_style="white-bg",
        hover_name="PCON24NM",
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


# Now create the graph that updates the constituency name based on hover and showing Years on x-axis and Display value
# of chosen dataframe on y-axis
@app.callback(
    Output("line_graph", "figure"),
    Input("mrp_map", "clickData"),
)
def create_graph(clickData):
    if clickData is None:
        constituency_name = "Aldershot"
    else:
        hover_text = clickData["points"][0].get("hovertext")
        constituency_name = hover_text if hover_text else "Aldershot"

    df = enriched_gdf.copy().query(f"PCON24NM == '{constituency_name}'")

    try:
        const_data = df[parties].iloc[0].to_dict()
    except KeyError:
        raise KeyError(df)

    vote_shares = pd.Series(const_data).rename("Vote Share").reset_index().rename({"index": "Party"}, axis="columns")

    fig = px.bar(vote_shares, x="Party", y="Vote Share", color="Party", color_discrete_map=colour_defs)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
