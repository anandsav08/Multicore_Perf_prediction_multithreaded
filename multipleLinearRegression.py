#
#	Regression idea from Kaggle.com
#
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

torch.manual_seed(42) # set a seed
k = 16
# kBestFeatures = ["L1-dcache-load-misses","L1-icache-loads","iTLB-loads","LLC-loads","L1-dcache-loads","dTLB-loads","instructions","LLC-load-misses","L1-icache-load-misses","branch-instructions","branch-loads","LLC-stores","context-switches","task-clock"]
# kBestFeatures = ["L1-dcache-loads","dTLB-loads","stalled-cycles-backend","instructions","cpu-cycles","branch-instructions","branch-loads","cpu-clock","task-clock","iTLB-load-misses","L1-icache-loads","iTLB-loads"]
kBestFeatures = ["L1-dcache-loads","dTLB-loads","instructions","stalled-cycles-backend","branch-instructions","branch-loads","cpu-cycles","cpu-clock","task-clock","iTLB-load-misses","L1-icache-loads","iTLB-loads","L1-dcache-load-misses","stalled-cycles-frontend"]
print("KbestFeatures length: ",len(kBestFeatures))
def normalize(df):
	normalized_df=(df - df.mean()) / df.std()
	df = normalized_df
	return df

def standardize(dataset):
	df_features = dataset
	dataset = (df_features - df_features.mean()) / df_features.std()
	df = dataset
	return df

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

def load_data():
	df = pd.read_csv("clean.csv")
	df = df.sample(frac=1).reset_index(drop=True)			# randomize samples
	y = df.loc[:,['time']].values							# We set 'time' column as our target 
	x_rest = df.loc[:,['threads']].values					# No need to normalize these fields, so taken separately. Later torch.cat() with rest normalized columns
	x_rest = torch.tensor(x_rest,dtype=float)
	x = df.loc[:, kBestFeatures].values						# Get K Best Features
	x = normalize(x)
	x = torch.tensor(x,dtype=float)
	x = torch.cat((x,x_rest),1)
	y = torch.tensor(y,dtype=float)
	return (x,y)



def MSE(y_predicted:torch.Tensor, y_target:torch.Tensor):
    error = y_predicted - y_target # element-wise substraction
    return torch.sum(error**2 ) / error.numel() # mean (sum/n)

class MultipleRegression(nn.Module):
    
    def __init__(self, n_weight:int):
        super(MultipleRegression, self).__init__()
        assert isinstance(n_weight, int)
        self.w = torch.randn(size =(n_weight,1), dtype=torch.double, 
                             requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def forward(self, X:torch.Tensor):
        assert isinstance(X, torch.Tensor)
        prediction = X @ self.w + self.b
        return prediction


(X,y) = load_data() 
print(X.shape)
print(y.shape)
train_size = int(0.92*len(X))
test_size = len(X) - train_size
X_train = X[0:train_size,:]
y_train = y[0:train_size,:]

X_test = X[train_size:,:]
y_test = y[train_size:,:]
print("x train shape: ",X_train.shape)
print("y train shape: ",y_train.shape)
print("x test shape: ",X_test.shape)
print("y test shape: ",y_test.shape)
model = MultipleRegression(n_weight = X.shape[1])

prediction = model(X)
MSE(y_predicted = prediction,y_target = y.reshape(-1,1))

import torch.nn.functional as F

F.mse_loss(prediction,y.reshape(-1,1))
optimizer = torch.optim.Adam([model.b,model.w],lr=0.001)

myMSE = list()
for epoch in tqdm(range(100000)):
	optimizer.zero_grad()
	predicted = model(X_train)
	loss = F.l1_loss(predicted,y_train.reshape(-1,1))
	myMSE.append(loss.item())
	loss.backward()
	optimizer.step()

plt.plot(myMSE)
plt.xlabel('Epoch (#)')
plt.ylabel('Mean Squared Error')
plt.show()


model.eval()
prediction = model(X_test)

y_pred = [pred.item() for pred in prediction ]
MSE_score = F.mse_loss(prediction,y_test.reshape(-1,1))

print("\n######### RESULTS #######")
print("MSE_score\t: %.4f"%(MSE_score.item()))
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


