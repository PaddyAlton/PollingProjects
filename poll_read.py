import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()
import pandas as pd

#  C://Users/Laptop/Documents/GitHub/polling_projects/poll_read.py 

#test = pd.read_excel("C://Users/Laptop/Documents/GitHub/polling_projects/test_excel1.xlsx")

data = np.loadtxt('C://Users/Laptop/Documents/Python_Scripts/polling4.txt', dtype=[('dates','10S'),('month','4S'),('pollster','10S'),('sample',float),('con',float),('lab',float),('ukip',float),('lib',float),('green',float),('others',float),('lead','5S')])

dlist = []

for ii, key in  enumerate(data.dtype.names): dlist.append(list(data[key]))

all_data = dict(list(zip(data.dtype.names,dlist)))

df = pd.DataFrame(all_data)

