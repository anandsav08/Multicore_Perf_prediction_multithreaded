import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]
kBestFeatures = ["L1-dcache-loads","dTLB-loads","instructions","stalled-cycles-backend","branch-instructions","branch-loads","cpu-cycles","cpu-clock","task-clock","iTLB-load-misses","L1-icache-loads","iTLB-loads","L1-dcache-load-misses","stalled-cycles-frontend"]
df = pd.read_csv("clean.csv")
df = df.sample(frac=1).reset_index(drop=True)			# randomize samples
X = df.loc[:, kBestFeatures].values 
X = (X - X.mean()) / X.std()
X_rest = np.array(df['threads'].values)
X_rest = np.expand_dims(X_rest,axis=1)
X = np.concatenate((X,X_rest),axis = 1)
y = df.loc[:, ['time']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


print(X_train.shape,y_train.shape)
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
y_pred = np.expand_dims(y_pred,axis=1)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n######### RESULTS #######")
print("MSE_score\t: %.4f"%(rmse))
r2 = r2_score(y_test,y_pred)
print("r2_score\t: %.4f "%(r2))
mae = mean_absolute_error(y_pred,y_test)
print("MAE\t\t: %.4f"%(mae))
print()
print("PREDICTED\tACTUAL")
for i in range(len(y_pred)):
	print("%.4f\t\t%.4f"%(y_pred[i],y_test[i]))
x_axis = np.linspace(1,len(y_pred),len(y_pred))
plt.plot(x_axis,y_pred,label="Predicted")
plt.plot(x_axis,y_test,label="Actual")
plt.legend()
plt.grid(visible=True,axis='both',color='r', linestyle='-', linewidth=0.1)
plt.show()

