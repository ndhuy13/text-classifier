import pandas as pd

check = pd.read_csv('check.csv')            #product_name, cat
false_result = pd.read_csv('final_false_result.csv')      #title	actual	pred_1	prob_1	pred_2	prob_2


new_cat = []
for index, row in false_result.iterrows():
    if check[check['product_name'] == row['title']].size != 0:
        new_cat.append(check[check['product_name'] == row['title']].iloc[0][2])
    else: new_cat.append("")

false_result['new_cat'] = new_cat
false_result.to_csv('adjusted.csv')



