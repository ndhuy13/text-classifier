

import pandas as pd

df = pd.DataFrame()

for i in range(2,22):
    df_temp = pd.read_csv('search_cat_tiki_{0}.csv'.format(i))
    df = pd.concat([df, df_temp])

df.to_csv('check.csv')

