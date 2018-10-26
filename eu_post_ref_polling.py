
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes()

eu_dat = pd.read_excel("EU_post_ref_polling.xlsx")

eu_dat.index = pd.to_datetime(
    eu_dat.Conducted.map(lambda x: x.split('-')[-1])
)

eu_dat['Right_e'] = 2.0*(
    eu_dat
        .Right
        .mul(1.0 - eu_dat.Right)
        .div(eu_dat.Sample)
        .map(np.sqrt)
)

eu_dat['Wrong_e'] = 2.0*(
    eu_dat
        .Wrong
        .mul(1.0 - eu_dat.Wrong)
        .div(eu_dat.Sample)
        .map(np.sqrt)
)

eu_dat_yg = eu_dat.query("Pollster=='YouGov'")

f, ax = plt.subplots()
f.set_tight_layout(True)

#eu_dat_yg.Wrong.plot(color='r', marker='o', ax=ax)

ax.errorbar(eu_dat_yg.index, eu_dat_yg.Wrong*100, eu_dat_yg.Wrong_e*100, fmt='ro', label='Wrong')

ax.errorbar(eu_dat_yg.index+pd.Timedelta(0.5,'d'), eu_dat_yg.Right*100, eu_dat_yg.Right_e*100, fmt='go', label='Right')

ax.legend(fontsize='large', ncol=2)

ax.set_title("Is the UK Right or Wrong to leave the EU?", fontsize='large')
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('%', fontsize='large')
ax.tick_params(labelsize=15)