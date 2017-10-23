# bokeh_example_map.py
# script for generating an interactive version of the EU referendum map,
# this time using Bokeh and a bit of manual work with the geoJSON file.
# PaddyAlton -- 2017-10-23


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import pandas as pd
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()
import json

import bokeh

from bokeh.io import show, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper,
    LinearColorMapper,
    ColorBar
)
from bokeh.palettes import Viridis6 as palette
from bokeh.palettes import RdBu8 as palette2
from bokeh.plotting import figure

palette.reverse()
palette2.reverse()

### Handling the constituency GeoJSON data:

data_dir = "C://Users/Laptop/Documents/GitHub/polling_projects/" # this may need changing to a directory of your choice.
wcon_geo = r""+data_dir+"Westminster_Parliamentary_Constituencies_December_2016_Ultra_Generalised_Clipped_Boundaries_in_Great_Britain.geojson" # the .json with boundaries and IDs

with open(wcon_geo) as f: # context manager for reading in the file
	geo_data = json.load(f) # read it into geo_data

# initialise lists to store the patch coordinates and constituency IDs
xcoords = []
ycoords = []
con_ids = []

for cnum in range(len(geo_data['features'])): # run through 632 constituencies, appending patch X and Y coordinates and IDs to the appropriate lists
	con_tag = geo_data['features'][cnum]['properties']['pcon16cd'] # get the constituency ID
	
	single_patch=False # sometimes, the constituencies aren't contiguous (e.g. multiple islands), so consist of multiple patches. These cases have to be handled carefully.
	if len(np.array(geo_data['features'][cnum]['geometry']['coordinates'][0]).shape)==2: single_patch=True
	
	if single_patch:
		coords = np.array(geo_data['features'][cnum]['geometry']['coordinates'][0]) # get the constituency patch coordinates
		xs, ys = coords[:,0], coords[:,1]
		
		xcoords.append(xs)
		ycoords.append(ys)
		con_ids.append(con_tag)
		
	else:
		n_patch = np.array(geo_data['features'][cnum]['geometry']['coordinates']).shape[0]
		for pn in range(n_patch): # we're going to break the constituency up into different patches and append all of them to the list
			coords = np.array(geo_data['features'][cnum]['geometry']['coordinates'][pn][0]) # get the constituency patch coordinates
			xs, ys = coords[:,0], coords[:,1]
			
			xcoords.append(xs)
			ycoords.append(ys)
			con_ids.append(con_tag)

### DATA
### the British Electoral Survey 2015 results data contains a tonne of demographic data as well as results from 2010 and 2015
BES = pd.read_excel(data_dir+"BES_2015_election_results.xlsx", sheetname='Data') # read BES data into a dataframe

### READ IN EU REFERENDUM RESULTS AND APPEND THEM TO THE BES DATAFRAME

euref=pd.read_excel(data_dir+"hanretty_EU_ref_figs.xlsx") # this has estimates (ref. C. Hanretty) for the EU referendum result in each constituency, with ONS constituency IDs

ref_results_series = np.empty(len(BES))

for xx,id in enumerate(BES.ONSConstID.values): ref_results_series[xx] = euref.loc[euref.PCON11CD==id].ref_result.values[0] # matching on ONS constituency ID

BES['Leave'] = ref_results_series*100. # -> percentage
BES['Remain']= (1. - ref_results_series)*100.

const_name   = [BES.ConstituencyName.loc[BES.ONSConstID==cid].values[0] for cid in con_ids] # get the constituency names
const_remain = [BES.Remain.loc[BES.ONSConstID==cid].values[0] for cid in con_ids] # get the constituency remain percentage

color_mapper = LinearColorMapper(palette=palette2, low=0, high=100)

### create Bokeh column data source

source = ColumnDataSource(data=dict(
    x=xcoords,
    y=ycoords,
    name=const_name,
    rem=np.round(const_remain,1)
))

# set up the figure

TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="EU Referendum Result", tools=TOOLS,
    x_axis_location=None, y_axis_location=None, plot_width=490, plot_height=750
)
p.grid.grid_line_color = None

p.patches('x', 'y', source=source,
          fill_color={'field': 'rem', 'transform': color_mapper},
          fill_alpha=0.85, line_color="white", line_width=0.5)

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("Remain", "@rem{'0,0'}%"),
    ("(Long, Lat)", "($x, $y)"),
]

color_bar = ColorBar(color_mapper=color_mapper, height=17, width=410, title='% REMAIN', title_standoff=8, title_text_font_size='10px',
                     label_standoff=5, border_line_color=None, location=(0,0), orientation='horizontal')

p.add_layout(color_bar, 'below')

output_file(data_dir+"Examples/interactive_referendum_map.html")
show(p)