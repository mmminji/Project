import pandas as pd
import os
from tqdm import trange

# result = pd.DataFrame(columns=['file','Day','Hour','Minute'])

# path = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/test'

# for file_name in os.listdir(path):

#     df = pd.read_csv(f'{path}/{file_name}')
#     temp = df[(df['Day'] == 5) & (df['TARGET']==0)][['Day','Hour','Minute']]
#     temp = temp.reset_index(drop=True)
#     temp_list = []
#     for i in range(len(temp)):
#         if temp['Minute'][i] == 0:
#             temp_list.append('{}_Day7_{}h{}0m'.format(file_name, temp['Hour'][i], temp['Minute'][i]))
#         else:
#             temp_list.append('{}_Day7_{}h{}m'.format(file_name, temp['Hour'][i], temp['Minute'][i]))
#     temp = pd.concat([pd.DataFrame(temp_list, columns=['file']), temp], axis=1)
#     result = pd.concat([result, temp])

#     temp = df[(df['Day'] == 6) & (df['TARGET']==0)][['Day','Hour','Minute']]
#     temp = temp.reset_index(drop=True)
#     temp_list = []
#     for i in range(len(temp)):
#         if temp['Minute'][i] == 0:
#             temp_list.append('{}_Day8_{}h{}0m'.format(file_name, temp['Hour'][i], temp['Minute'][i]))
#         else:
#             temp_list.append('{}_Day8_{}h{}m'.format(file_name, temp['Hour'][i], temp['Minute'][i]))
#     temp = pd.concat([pd.DataFrame(temp_list, columns=['file']), temp], axis=1)
#     result = pd.concat([result, temp])

# result.reset_index(drop=True, inplace=True)
# result.to_csv('temp.csv', index=False)


def after1(df):
    result = pd.read_csv('temp.csv')

    col_list = df.columns.difference(['id'])

    for i in trange(len(result)):
        index = df[df['id'] == result['file'][i]].index.values[0]
        for col in col_list:
            df[col][index] = 0

    return df


def after2(df):
    result = pd.read_csv('temp.csv')

    col_list = df.columns.difference(['id'])

    for i in trange(len(df)):
        for j in df.columns.difference(['id']):
            if df[j][i] < 0:
                df[j][i] = (df[j][i-1] + df[j][i+1])/2

    for i in trange(len(result)):
        index = df[df['id'] == result['file'][i]].index.values[0]
        for col in col_list:
            df[col][index] = 0
            
    return df

# df = pd.read_csv('submission/RF_10000.csv')
# df = after1(df)
# df.to_csv('submission/RF_10000_aft1.csv', index=False)