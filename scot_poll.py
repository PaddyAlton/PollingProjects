##### scot_poll.py
##### a python script for reading in the Scottish independence polling data and visualising it in a useful way
##### PaddyAlton -- 2017/10/17 (v1.0)

### IMPORT REQUIRED MODULES

import numpy as np
import matplotlib.pyplot as plt; plt.ion() # interactive mode plotting
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()
import pandas as pd


### READ IN DATAFRAMES


def dt_convert(item):
	
	""" 
	Custom converter function for reading in the dates of polls.
	(the problem is, polls with a single date marker get correctly read in as datetimes, whereas those conducted 
	over a number of days do not. So: convert them all to strings, god will recognise his own...)
	"""
	
	splitup = str(item).split('-')
	
	if len(splitup)==2: return splitup[-1]
	
	else: return str(item).split(' ')[0]

scot_data = pd.read_excel("C://Users/Laptop/Documents/GitHub/polling_projects/Scots_independence_polls.xlsx", converters={'Date':dt_convert})


dates = scot_data.Date.values
years = scot_data.Year.values

for ii in range(len(dates)):
	if dates[ii].endswith(str(years[ii])):
		pass
	else: dates[ii] += ' '+str(years[ii])

scot_data.drop('Year', axis=1, inplace=True)
scot_data['Date'] = pd.to_datetime(dates)

scot_data.set_index('Date', inplace=True)

scot_data.Pollster = scot_data.Pollster.str.split('/',expand=True)[0].values # this command drops the commissioning organisation leaving only the polling organisation itself.

scot_data['yes_err'] = np.sqrt(((1.-scot_data.Yes)*scot_data.Yes)/scot_data.N_sample)
scot_data['no_err']  = np.sqrt(((1.-scot_data.No)*scot_data.No)/scot_data.N_sample)

### NOW PLOT:

plt.errorbar(scot_data.index, scot_data.Yes*100., yerr=scot_data.yes_err*100., fmt='go', label='Yes', alpha=0.9)
plt.errorbar(scot_data.index, scot_data.No*100.,  yerr=scot_data.no_err*100.,  fmt='ro', label='No',  alpha=0.9)

plt.legend(ncol=2, fontsize=20)


### TRY OUT SOME ROLLING WINDOW STUFF:

roll_yes = scot_data.rolling(14, min_periods=1, win_type=('gaussian',7)).mean().Yes*100.
roll_no  = scot_data.rolling(14, min_periods=1, win_type=('gaussian',7)).mean().No*100.

roll_yerr = scot_data.rolling(14, min_periods=1, win_type=('gaussian',7)).mean().yes_err*100.
roll_nerr = scot_data.rolling(14, min_periods=1, win_type=('gaussian',7)).mean().no_err*100.

plt.plot(scot_data.index, roll_yes, 'g-', linewidth=2)
plt.plot(scot_data.index, roll_no,  'r-', linewidth=2)

plt.fill_between(scot_data.index, y1= roll_yes - roll_yerr, y2 = roll_yes + roll_yerr, color='g', alpha=0.3)
plt.fill_between(scot_data.index, y1= roll_no - roll_nerr,  y2 = roll_no + roll_nerr,  color='r', alpha=0.3)

plt.fill_between(scot_data.index, y1= roll_yes - 2*roll_yerr, y2 = roll_yes + 2*roll_yerr, color='g', alpha=0.3)
plt.fill_between(scot_data.index, y1= roll_no - 2*roll_nerr,  y2 = roll_no + 2*roll_nerr,  color='r', alpha=0.3)



### PLOT REF RESULT:

mask = scot_data.Pollster=='Referendum' # mark the actual election result
ref_info = scot_data.loc[mask]

plt.plot(ref_info.index, ref_info.Yes*100., 'ks', markersize=18)
plt.plot(ref_info.index, ref_info.No*100.,  'ks', markersize=18)

plt.plot(ref_info.index, ref_info.Yes*100., 'gs', markersize=12)
plt.plot(ref_info.index, ref_info.No*100.,  'rs', markersize=12)

plt.tight_layout()
plt.tick_params(labelsize=20)
plt.title('Scottish Independence Polling Pre/Post-Referendum', fontsize=22)
plt.xlabel('Date', fontsize=22)
plt.ylabel('Percentage Support', fontsize=22)