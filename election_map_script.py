# election_map_script.py
# this is a script for plotting data broken down by UK parliamentary constituency. Uses folium to make a nice , zoomable map that you can open in a browser.
# PaddyAlton -- 2017-10-19

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()
import pandas as pd
import folium

data_dir = "C://Users/Laptop/Documents/GitHub/polling_projects/" # this may need changing to a directory of your choice.

### the British Electoral Survey 2015 results data contains a tonne of demographic data as well as results from 2010 and 2015
BES = pd.read_excel(data_dir+"BES_2015_election_results.xlsx", sheetname='Data') # read BES data into a dataframe

### READ IN EU REFERENDUM RESULTS AND APPEND THEM TO THE BES DATAFRAME

euref=pd.read_excel(data_dir+"hanretty_EU_ref_figs.xlsx") # this has estimates (ref. C. Hanretty) for the EU referendum result in each constituency, with ONS constituency IDs

ref_results_series = np.empty(len(BES))

for xx,id in enumerate(BES.ONSConstID.values): ref_results_series[xx] = euref.loc[euref.PCON11CD==id].ref_result.values[0] # matching on ONS constituency ID

BES['Leave'] = ref_results_series*100. # -> percentage
BES['Remain']= (1. - ref_results_series)*100.

### MAKE THE MAP

wcon_geo = r""+data_dir+"Westminster_Parliamentary_Constituencies_December_2016_Ultra_Generalised_Clipped_Boundaries_in_Great_Britain.geojson" # the .json with boundaries and IDs

test_map = folium.Map(location=[55,3.5], zoom_start=3, tiles='Mapbox Bright') # make the map object

test_map.choropleth(wcon_geo, data=BES, fill_color='RdBu', columns=['ONSConstID','Remain'], threshold_scale=[20,35,50,65,80], key_on='feature.properties.pcon16cd', legend_name='EU referendum % Remain')
# boundaries come from the .json, data from the BES dataframe, columns=[ID to match on, data column to plot]

test_map.save(data_dir+'Examples/referendum_map.html') # create output