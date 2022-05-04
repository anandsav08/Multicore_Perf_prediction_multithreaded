import pandas as pd
import numpy as np
import torch

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]
# kBestFeatures = ["L1-dcache-loads","dTLB-loads","stalled-cycles-backend","instructions","cpu-cycles","branch-instructions","branch-loads","cpu-clock","task-clock","iTLB-load-misses","L1-icache-loads","iTLB-loads"]
params = {"objective":"reg:linear",'colsample_bytree': 0.5,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}


def normalize(df):
	normalized_df=(df - df.mean()) / df.std()
	df = normalized_df
	return df

def load_data():
	df = pd.read_csv("clean.csv")
	y = df.loc[:,['time']]							# We set 'time' column as our target 
	x_rest = df.loc[:,['threads']]					# No need to normalize these fields, so taken separately. Later torch.cat() with rest normalized columns\

	x = df.loc[:, kBestFeatures]						# Get K Best Features
	x = normalize(x)
	x = pd.concat([x,x_rest],axis = 1)
	return (x,y)

(X,y) = load_data()
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.6, learning_rate = 0.2,
                max_depth = 5, alpha = 10, n_estimators = 50)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
y_pred = preds
y_test = np.squeeze(y_test,axis=1)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("\n######### RESULTS #######")
print("MSE_score\t: %.4f"%(rmse))
r2 = r2_score(y_test,y_pred)
print("r2_score\t: %.4f "%(r2))
mae = mean_absolute_error(y_pred,y_test)
print("MAE\t\t: %.4f"%(mae))
print()

x_axis = np.linspace(1,len(preds),len(preds))
plt.plot(x_axis,preds,label="Predicted")
plt.plot(x_axis,y_test,label="Actual")
plt.legend()
plt.grid(visible=True,axis='both',color='r', linestyle='-', linewidth=0.1)
plt.show()
# k-fold Cross Validation using XGBoost
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=6,
                    num_boost_round=500,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
res = cv_results
print("After k-fold cross validation: ")
print((cv_results["test-rmse-mean"]))

res.plot()
plt.show()


