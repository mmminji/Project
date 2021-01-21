from LGBM_data_prepro import *
from after_process import *
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)


categorical_columns = []
categorical_dims =  {}
for col in X_train_1.nunique()[X_train_1.nunique()==2].index:
    # print(col, X_train_1[col].nunique())
    l_enc = LabelEncoder()
    # X_train_1[col] = X_train_1[col].fillna("VV_likely")
    X_train_1[col] = l_enc.fit_transform(X_train_1[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

# for col in X_train_1.columns[X_train_1.dtypes == 'float64']:
#     X_train_1.fillna(X_train_1.loc[train_indices, col].mean(), inplace=True)

unused_feat = ['Set']
target = 'TARGET'

features = [ col for col in X_train_1.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# define your embedding sizes : here just a random choice
cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]*8
del cat_emb_dim[-1]
# len(cat_emb_dim)


from pytorch_tabnet.tab_model import TabNetRegressor

# clf = TabNetRegressor()
clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

X_train_1 = X_train_1.values
X_valid_1 = X_valid_1.values
Y_train_1 = Y_train_1.values.reshape(-1,1)
Y_valid_1 = Y_valid_1.values.reshape(-1,1)

max_epochs = 1000

clf.fit(
    X_train=X_train_1, y_train=Y_train_1,
    eval_set=[(X_train_1, Y_train_1), (X_valid_1, Y_valid_1)],
    eval_name=['train', 'valid'],
    eval_metric=['mae', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    # batch_size=1024,
    # virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)


X_test = X_test.values
preds = clf.predict(X_test)
preds[:48]

sub = pd.DataFrame(preds)

sub['1'] = sub[0]*0.6
sub['2'] = sub[0]*0.7
sub['3'] = sub[0]*0.8
sub['4'] = sub[0]*0.9
sub['5'] = sub[0]*1
sub['6'] = sub[0]*1.1
sub['7'] = sub[0]*1.2
sub['8'] = sub[0]*1.3
sub['9'] = sub[0]*1.4
sub[:48]



# clf2 = TabNetRegressor()
clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

X_train_2 = X_train_2.values
X_valid_2 = X_valid_2.values
Y_train_2 = Y_train_2.values.reshape(-1,1)
Y_valid_2 = Y_valid_2.values.reshape(-1,1)

max_epochs = 1000

clf2.fit(
    X_train=X_train_2, y_train=Y_train_2,
    eval_set=[(X_train_2, Y_train_2), (X_valid_2, Y_valid_2)],
    eval_name=['train', 'valid'],
    eval_metric=['mae', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    # batch_size=1024,
    # virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)


X_test = X_test.values
preds2 = clf2.predict(X_test)

sub2 = pd.DataFrame(preds2)

sub2['1'] = sub2[0]*0.6
sub2['2'] = sub2[0]*0.7
sub2['3'] = sub2[0]*0.8
sub2['4'] = sub2[0]*0.9
sub2['5'] = sub2[0]*1
sub2['6'] = sub2[0]*1.1
sub2['7'] = sub2[0]*1.2
sub2['8'] = sub2[0]*1.3
sub2['9'] = sub[0]*1.4


df = pd.DataFrame()
for i in range(81):
    df = pd.concat([df, sub[i*48:(i+1)*48]])
    df = pd.concat([df, sub2[i*48:(i+1)*48]])
df.to_csv('tabnet.csv')