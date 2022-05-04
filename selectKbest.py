import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_regression
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

global df
global target
def selectBestFeatures(n):
	global df
	global target
	df = pd.read_csv("clean.csv")
	y = df['time']
	target = df['time']
	df1 = df[df.columns[1:-4]]
	df1 = (df1-df1.mean())/df1.std()
	df2 = df['threads']
	df = pd.concat([df1,df2],axis=1)
	x = df	
	bestfeatures = SelectKBest(score_func=sklearn.feature_selection.f_regression,k='all')
	fit = bestfeatures.fit(x,y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(x.columns)

	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['Specs','Score']
	return featureScores

def showHeatMap(features):
	global df,target
	df = df [features]
	df = pd.concat([df,target],axis = 1)
	print(df)
	f, ax = plt.subplots(figsize=(60, 10))
	corr = df.corr()
	hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.1f',
	                 linewidths=.05)
	f.subplots_adjust(top=0.93)
	t= f.suptitle(' Correlation Heatmap', fontsize=10)
	plt.show()

def get_k_best_features_list():
	k = 14
	kBestFeatures = selectBestFeatures(k)
	result = kBestFeatures.nlargest(k,'Score')
	return result
	

if __name__ == "__main__":
	features = get_k_best_features_list()
	kBestFeatures = features['Specs']
	print("k Best Features:\n",kBestFeatures)
	showHeatMap(kBestFeatures)