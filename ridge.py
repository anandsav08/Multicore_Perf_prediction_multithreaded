import pandas as pd
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]
kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]
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
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = RidgeCV(alphas=arange(0, 1, 10.0), cv=cv, scoring='r2')

#fit model
model.fit(X_train, y_train)

#display lambda that produced the lowest test MSE
print(model.alpha_)
y_pred = model.predict(X_test)

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

#Plot graph
x_axis = np.linspace(1,len(y_pred),len(y_pred))
plt.plot(x_axis,y_pred,label="Predicted",linestyle='--')
plt.plot(x_axis,y_test,label="Actual")
plt.legend()
plt.grid(visible=True,axis='both',color='r', linestyle='-', linewidth=0.1)
plt.show()