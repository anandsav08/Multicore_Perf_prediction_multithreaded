import pandas as pd
from numpy import arange
from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error

kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]

# DATA section
df = pd.read_csv("clean.csv")
df = df.sample(frac=1).reset_index(drop=True)
X = df[kBestFeatures]
X = (X - X.mean()) / X.std()
X_rest =df['threads']
X = pd.concat([X,X_rest],axis=1)
y = df['time']

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08)

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X_train, y_train)
# print(model.alpha_)

# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))

y_pred = lasso_best.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n######### RESULTS #######")
print("MSE_score\t: %.4f"%(rmse))
r2 = r2_score(y_test,y_pred)
print("r2_score\t: %.4f "%(r2))
mae = mean_absolute_error(y_pred,y_test)
print("MAE\t\t: %.4f"%(mae))
print()
print("PREDICTED: ")
print(abs(y_pred))
print("\nACTUAL")
print(y_test)

# GRAPH section
x_axis = np.linspace(1,len(y_pred),len(y_pred))
plt.plot(x_axis,y_pred,label="Predicted",linestyle='--')
plt.plot(x_axis,y_test,label="Actual")
plt.legend()
plt.grid(visible=True,axis='both',color='r', linestyle='-', linewidth=0.1)
plt.show()
