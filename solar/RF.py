from LGBM_data_prepro import *
from after_process import *
from tqdm import tqdm

X_train_1 = df_train.iloc[:, :-2]
Y_train_1 = df_train.iloc[:, -2]

X_train_2 = df_train.iloc[:, :-2]
Y_train_2 = df_train.iloc[:, -1]

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10000, random_state=42,min_samples_split=10)
rf.fit(X_train_1, Y_train_1)

# Get the predictions of all trees for all observations
# Each observation has N predictions from the N trees
pred_Q = pd.DataFrame()
for pred in tqdm(rf.estimators_):
    temp = pd.Series(pred.predict(X_test).round(2))
    pred_Q = pd.concat([pred_Q,temp],axis=1)
# pred_Q.head()

RF_actual_pred_1 = pd.DataFrame()

for q in tqdm(quantiles):
    s = pred_Q.quantile(q=q, axis=1)
    RF_actual_pred_1 = pd.concat([RF_actual_pred_1,s],axis=1,sort=False)
   
RF_actual_pred_1.columns=quantiles
# RF_actual_pred_1['actual'] = Y_test
RF_actual_pred_1['interval'] = RF_actual_pred_1[np.max(quantiles)] - RF_actual_pred_1[np.min(quantiles)]
# RF_actual_pred_1 = RF_actual_pred_1.sort_values('interval')
RF_actual_pred_1 = RF_actual_pred_1.round(2)
RF_actual_pred_1.iloc[:, :-1]

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = RF_actual_pred_1.iloc[:, :-1].sort_index().values


rf = RandomForestRegressor(n_estimators=10000, random_state=42,min_samples_split=10)
rf.fit(X_train_1, Y_train_1)

# Get the predictions of all trees for all observations
# Each observation has N predictions from the N trees
pred_Q = pd.DataFrame()
for pred in tqdm(rf.estimators_):
    temp = pd.Series(pred.predict(X_test).round(2))
    pred_Q = pd.concat([pred_Q,temp],axis=1)
# pred_Q.head()

RF_actual_pred_2 = pd.DataFrame()

for q in tqdm(quantiles):
    s = pred_Q.quantile(q=q, axis=1)
    RF_actual_pred_2 = pd.concat([RF_actual_pred_2,s],axis=1,sort=False)
   
RF_actual_pred_2.columns=quantiles
# RF_actual_pred_2['actual'] = Y_test
RF_actual_pred_2['interval'] = RF_actual_pred_2[np.max(quantiles)] - RF_actual_pred_2[np.min(quantiles)]
# RF_actual_pred_2 = RF_actual_pred_2.sort_values('interval')
RF_actual_pred_2 = RF_actual_pred_2.round(2)
RF_actual_pred_2[:48]


submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = RF_actual_pred_2.iloc[:, :-1].sort_index().values

submission.to_csv('submission/RF_10000.csv', index=False)