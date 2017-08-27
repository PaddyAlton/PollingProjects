##### poll_read.py
##### a python script for reading in the polling data and visualising it in a useful way
##### PaddyAlton -- 2017/08/27 (v1.0)

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

# reads in all four Excel sheets as a python dictionary
test = pd.read_excel("C://Users/Laptop/Documents/GitHub/polling_projects/UK_polling_2005_to_present.xlsx", sheetname=None, converters={'date conducted':dt_convert}) 

# extract each DataFrame and name it appropriately:
polls2010 = test['For 2010'] # polling from 2005 - 2010
polls2015 = test['For 2015'] # polling from 2010 - 2015
polls2017 = test['For 2017'] # polling from 2015 - 2017
polls2022 = test['For 2022'] # polling from 2017 -->


### DEFINE SOME NEW FUNCTIONS FOR DATAFRAME JIGGERY-POKERY

def dt_combine(df):
	
	"""
	Input: a DataFrame
	Returns: a list of dates in a common format (dd mm yyyy)
	Creates a common format for the dates, pulling in the 'year conducted' column and extracting the day and month from
	the 'date conducted' column.
	"""
	
	year = list(df['year conducted'])
	date = list(df['date conducted'])
	
	for ii, item in enumerate(date):
		if len(item.split('-'))!=3: date[ii] = item+' '+str(year[ii])
		else: date[ii] = item.split('-')[2]+' '+item.split('-')[1]+' '+str(year[ii])
	return date

def dt_fix(df):
	
	"""
	Input: a DataFrame
	Returns: a new DataFrame with a datetime-index 
	Passes the input to dt_combine(), converts the returned list to datetime format, stripping out any leading whitespace,
	and uses as the DataFrame index. Then throws out the original year/date conducted columns.
	(oh, and for good measure it ensures all sample sizes are in numeric format)
	"""
	
	df['conducted'] = dt_combine(df)
	df.conducted = df.conducted.str.strip()
	df.conducted = [pd.to_datetime(entry,dayfirst=True,errors='coerce') for entry in df.conducted]
	
	df['Sample size'] = pd.to_numeric(df['Sample size'],errors='coerce') # 
	
	return df[df.columns[2:]].set_index('conducted')

month_dict1 = {'Jan':0,'Feb':31,'Mar':59,'Apr':90,'May':120,'June':151,'Jun':151,'Jul':181,'Aug':212,'Sep':243,'Oct':273,'Nov':304,'Dec':334}
month_dict2 = {1:0,2:31,3:59,4:90,5:120,6:151,7:181,8:212,9:243,10:273,11:304,12:334} # define this for calculating fraction of a year from month/day combos

def frac_year(df):
	
	"""
	Input: a DataFrame with a datetime index
	Returns: a NumPy array with the dates converted to fractional A.D. years.
	"""
	
	conducted = df.index
	days = np.array([item.year-2005 for item in conducted])*365.25 + np.array([month_dict2[item.month] for item in conducted]) + np.array([item.day for item in conducted])
	return 2005+days/365.25
# plot scatter points on top (hence separate loop)

### APPLY THOSE FUNCTIONS TO THE DATA

polls2010 = dt_fix(polls2010)
polls2015 = dt_fix(polls2015)
polls2017 = dt_fix(polls2017)
polls2022 = dt_fix(polls2022)

polls2010['frac_year'] = frac_year(polls2010)
polls2015['frac_year'] = frac_year(polls2015)
polls2017['frac_year'] = frac_year(polls2017)
polls2022['frac_year'] = frac_year(polls2022)

polls2022.fillna(method='bfill',limit=5,inplace=True) # this takes columns which are NaN and backfills with later observations (up to a limit of five missing observations)
polls2017.fillna(method='bfill',limit=5,inplace=True)
polls2015.fillna(method='bfill',limit=5,inplace=True)


### PERFORM AN OUTER JOIN ON THE DATA TO CREATE ONE DATAFRAME

polls = pd.concat([polls2022,polls2017,polls2015,polls2010],join='outer') # keeps all columns, fills in missing values as NaN.

parties = list(polls.columns[[0,3,2,10,1,8,6,5,4]]) # list of party names (in a preferred ordering)


### CONVERT RESULTS TO PERCENTAGES

def p_fix(pd_series): return pd.to_numeric(pd_series, errors='coerce') * 100.

for pp in parties: polls[pp] = p_fix(polls[pp])


### PLOTTING COMMANDS

_ = parties.pop(-1) # remove lead
_ = parties.pop(-1) # remove Others

c_dict = {'Con':'b','Lab':'r','LD':'Orange','UKIP':'Indigo','Green':'g','SNP':'y','Plaid':'darkgreen'} # plotting colours

f,ax=plt.subplots() # initialise plot axis
f.set_tight_layout(True)

#### PLOTTING: LOOP OVER PARTIES

def running_average(time,result,window=14/365.25):
	""" This function creates a running average based on the time-series (time [frac_year], result) with window [default = 1 week]"""
	output = np.empty_like(time)
	for tt in range(time.size): output[tt] = np.mean(result[np.abs(time-time[tt])<=window])
	return output

def formal_error(df, party):
	"""
	Inputs: polls DataFrame, chosen party
	Returns: formal statistical (1-sigma) uncertainty based on sample size
	"""
	ydat = df[party].values/100.
	return 100.*np.sqrt(ydat*(1.-ydat)/df["Sample size"])

for part in parties: # loop over list of parties
	xdat = polls.frac_year
	rav = running_average(polls.frac_year, polls[part])
	conf = 100.*np.sqrt((rav/100.)*(1-rav/100.)/2000.) # calculate 1-sigma confidence interval assuming sample size ~ 2000
	
	ax.plot(xdat, rav, '-', color=c_dict[part], linewidth=3, label=part) # plot running average for current party
	ax.fill_between(xdat,rav-conf,rav+conf, color=c_dict[part], alpha=0.3) # plot 1-sigma confidence interval
	ax.fill_between(xdat,rav-conf-conf,rav+conf+conf, color=c_dict[part], alpha=0.3) # plot 2-sigma confidence interval

# plot scatter points on top (hence separate loop)
#for part in parties: ax.scatter(polls.frac_year, polls[part].values, color=c_dict[part], alpha=0.3)

# plot points with formal errorbars attached
for part in parties: ax.errorbar(polls.frac_year, polls[part].values, yerr=formal_error(polls, part), fmt='o', color=c_dict[part], alpha=0.3)

### mark important dates:

elections = ['5 May 2005','6 May 2010','7 May 2015', '22 June 2016', '8 June 2017']
elections_fy = [float(dd.split(' ')[2])+(float(month_dict1[dd.split(' ')[1]])+float(dd.split(' ')[0]))/365.25 for dd in elections] # convert to fractional years A.D.

brexit = elections_fy.pop(3) # pop out Brexit from the elections list (so we can use a different colour)

ax.axvline(brexit, linestyle='--',color='r',linewidth=3)
for election in elections_fy: ax.axvline(election,linestyle='--',color='k',linewidth=3)

### labelling and legend commands

ax.legend(fontsize=22, loc=0, ncol=len(parties))
ax.set_title('Political polling 2005-'+str(polls.frac_year[0]).split('.')[0], fontsize=22)
ax.set_xlabel('Year',fontsize=22)
ax.set_ylabel('% share', fontsize=22)
ax.tick_params(labelsize=20)

ax.set_xlim(polls.frac_year.iloc[-1]-0.01,polls.frac_year.iloc[0])
ax.set_ylim(0,55.)



plt.show() # (only does anything if you switch interactive plotting off)