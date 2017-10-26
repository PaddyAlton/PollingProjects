# elections_ML.py
# an experiment with SciKit-Learn and election data and demography
# PaddyAlton -- 2017-10-25

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()
import pandas as pd

from bokeh.io import show, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
	CategoricalColorMapper,
	LinearAxis)

from bokeh.plotting import figure


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


################ GET THE DATA ############

data_dir = "C://Users/Laptop/Documents/GitHub/polling_projects/" # this may need changing to a directory of your choice.

### the British Electoral Survey 2015 results data contains a tonne of demographic data as well as results from 2010 and 2015
BES = pd.read_excel(data_dir+"BES_2015_election_results.xlsx", sheetname='Data') # read BES data into a dataframe

### READ IN EU REFERENDUM RESULTS AND APPEND THEM TO THE BES DATAFRAME

euref=pd.read_excel(data_dir+"hanretty_EU_ref_figs.xlsx") # this has estimates (ref. C. Hanretty) for the EU referendum result in each constituency, with ONS constituency IDs

ref_results_series = np.empty(len(BES))

for xx,id in enumerate(BES.ONSConstID.values): ref_results_series[xx] = euref.loc[euref.PCON11CD==id].ref_result.values[0] # matching on ONS constituency ID

BES['Remain']= (1. - ref_results_series)*100. # -> percentage

############### THE IDEA: CAN WE PREDICT BES.Remain FROM THE PANOPLY OF DEMOGRAPHIC DATA IN BES? ##############

BES.drop(np.where(BES.ConstituencyName=='Buckingham')[0][0], inplace=True) # drop the Speaker's constituency

mBES = BES.copy() # we're going to modify the BES data in a number of ways:

# first, drop a tonne of columns
mBES.drop(BES.columns[:6],axis=1, inplace=True) # drop leading columns (constituency IDs, Names, categorical data)
mBES.drop([x for x in mBES.columns if x.endswith('10')],axis=1,inplace=True) # no results from 2010
mBES.drop([x for x in mBES.columns if x.endswith('1015')],axis=1,inplace=True) # no swing data (this might actually be an interesting one to look at in future)
mBES.drop([x for x in mBES.columns if x.endswith('Vote15')],axis=1,inplace=True) # no need for raw count
mBES.drop([x for x in mBES.columns if x.startswith('Winner')],axis=1,inplace=True) # drop categorical data
mBES.drop([x for x in mBES.columns if x.startswith('Majority')],axis=1,inplace=True) # no need for raw count
mBES.drop([x for x in mBES.columns if x.startswith('Electorate')],axis=1,inplace=True) # no need for raw count
mBES.drop([x for x in mBES.columns if x.startswith('ConPPC')],axis=1,inplace=True) # drop categorical data
mBES.drop([x for x in mBES.columns if x.startswith('LabPPC')],axis=1,inplace=True) # drop categorical data
mBES.drop([x for x in mBES.columns if x.startswith('LabPCC')],axis=1,inplace=True) # drop categorical data # NOTE TYPO IN COLUMN HEADER
mBES.drop([x for x in mBES.columns if x.startswith('LDPCC')],axis=1,inplace=True) # drop categorical data # NOTE TYPO IN COLUMN HEADER
mBES.drop([x for x in mBES.columns if x.startswith('LDPPC')],axis=1,inplace=True) # drop categorical data 
mBES.drop([x for x in mBES.columns if x.startswith('UKIPPP')],axis=1,inplace=True) # drop categorical data # NOTE TYPO IN COLUMN HEADER - UKIPPPPC
mBES.drop([x for x in mBES.columns if x.startswith('SNPPPC')],axis=1,inplace=True) # drop categorical data
mBES.drop([x for x in mBES.columns if x.startswith('PCPPC')],axis=1,inplace=True) # drop categorical data
mBES.drop([x for x in mBES.columns if x.startswith('GreenPPC')],axis=1,inplace=True) # drop categorical data
mBES.drop('c11Population',axis=1,inplace=True)

# next, let's impute some NaN values
mBES['SNP15'].loc[mBES.SNP15.isnull()] = 0. # BES give NaN if a party didn't stand, but this just means they got 0 votes.
mBES['PC15'].loc[mBES.PC15.isnull()] = 0.
mBES['UKIP15'].loc[mBES.UKIP15.isnull()] = 0.
mBES['Green15'].loc[mBES.Green15.isnull()] = 0.
mBES['Other15'].loc[mBES.Other15.isnull()] = 0.

# lots of c11 columns have no scotland data. Now, we should be able to tell from some of the variab;es whether we're in Scotland, so no need to drop all the
# Scottish data, or all these columns. For now we'll just put in a dummy value, the median across rUK. A more sophisticated approach might include
# looking for correlations in each column with data that DOES exist.

cols_with_noscotdata = [col for col in mBES.columns if np.sum(mBES[col].isnull())>1]

for col in cols_with_noscotdata: mBES[col].loc[mBES[col].isnull()] = mBES[col].median()

feature_names = mBES.columns[:-1] # NB 'Remain' is the target variable, we exclude it here

### Now, mBES ought to be prepared. Lots of numerical columns, no NaN data, ought to be able to use Lasso on it...

X_full = mBES.drop('Remain',axis=1).values # grid of feature values (dropping the target variable, obviously)
Y_full = mBES.Remain.values # target variable array of values

# split the data into a training set and a test set:
X_train, X_test, Y_train, Y_test, nam_train, nam_test = train_test_split(X_full, Y_full, BES.ConstituencyName.values, test_size=0.5, random_state=9)

model = Lasso() # create a Lasso regression model. The regularisation is important as it prevents overfitting.

model.fit(X_train,Y_train) # fit the model

plt.plot(Y_train, model.predict(X_train),'ro', label='Training set')
plt.plot(Y_test,  model.predict(X_test), 'bo', label='Test set')
plt.legend(fontsize=20, loc=2)
plt.tick_params(labelsize=18)

plt.title('Lasso regression: predicting % Remain from demographic and electoral data', fontsize=20)
plt.xlabel('Estimates of actual % Remain', fontsize=20)
plt.ylabel('Predicted % Remain from our model', fontsize=20)

plt.tight_layout()

########

# interactive plot:

source = ColumnDataSource(data=dict(
	x=list(np.concatenate((Y_train,Y_test)) ),
	y=list( np.concatenate((model.predict(X_train),model.predict(X_test))) ),
	type=np.concatenate((['train' for ii in range(len(X_train))],['test' for ii in range(len(X_test))])),
	name=np.concatenate((nam_train, nam_test))
))

TOOLS = "pan,wheel_zoom,reset,hover,save"

mapper = CategoricalColorMapper(palette=["firebrick","dodgerblue"], factors=["train","test"])

p = figure(title="EU Referendum Regression", tools=TOOLS, x_axis_location=None, y_axis_location=None)

p.circle('x', 'y', source=source, color={'field': 'type', 'transform': mapper}, alpha=0.8, size=12, legend='type')

p.legend.location = "top_left"
p.legend.label_text_font_size = "20pt"
p.legend.orientation = "horizontal"

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("Remain", "@x{'0,0'}%"),
    ("Model Prediction", "@y{'0,0'}%")
]

xaxis = LinearAxis(axis_label="Constituency Referendum Result (%)")
yaxis = LinearAxis(axis_label="Model Predicted Result (%)")
p.add_layout(xaxis, 'below')
p.add_layout(yaxis, 'left')

output_file(data_dir+"Examples/EU_ref_regression.html")
show(p)

########

feature_names_r = feature_names[model.coef_ != 0]
key_features = feature_names_r[np.argsort(np.abs(model.coef_[model.coef_ != 0]))][::-1]
key_coef = model.coef_[model.coef_ != 0][np.argsort(np.abs(model.coef_[model.coef_ != 0]))][::-1]

for feat, coef in zip(key_features,key_coef): print feat, coef
