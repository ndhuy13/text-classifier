import pandas as pd

df = pd.read_csv('final_result.csv')
true_1 = 0
true_2 = 0
no_recomendation = 0

# for index, row in df.iterrows():
#     if row['actual'] == row['pred_1']:
#         if row['prob_1'] > 0.7:
#             true_1 += 1
#         else: no_recomendation += 1
#         continue
#     if row['actual'] == row['pred_2']:
#         if row['prob_1'] > 0.5:
#             true_2 += 1
#         else: no_recomendation += 1

for index, row in df.iterrows():
    if row['actual'] == row['pred_1']:
        true_1 += 1
        continue
    if row['actual'] == row['pred_2']:
        true_2 += 1

num_of_row = df.shape[0]


df_false = df[df['actual'] != df['pred_1']]
df_false = df_false[df_false['actual'] != df_false['pred_2']]
df_false.to_csv('final_false_result.csv')

print('Accuracy_1:', true_1/(num_of_row ))
print('Accuracy_2:', (true_2 + true_1)/(num_of_row))
# print('No Recomendation:', no_recomendation/num_of_row)