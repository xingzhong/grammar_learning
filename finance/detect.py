import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn import tree
from sklearn.svm import LinearSVC
import shutil
import os
import matplotlib.pyplot as plt
import pickle

def trainModel():
	truths = pd.read_csv('truth.csv', header=None, index_col=0, prefix='Y')
	features = pd.read_csv("data.csv", header=None, index_col=0, prefix='X')
	data = features.join(truths)
	data['X5'] = np.sqrt((data.X2 - data.X3)**2)
	data['X6'] = data.X2 * data.X3
	data['X7'] = data.X2 - data.X3
	data0 = data[data.X1>0]
	data1 = data[data.X1<0]
	labeled0 = data0[~data0.Y1.isnull() ]
	labeled1 = data1[~data1.Y1.isnull() ]
	X0 = labeled0[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
	Y0 = labeled0.Y1.values
	X1 = labeled1[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
	Y1 = labeled1.Y1.values
	m0 = OneVsRestClassifier(GaussianNB())
	m1 = OneVsRestClassifier(GaussianNB())
	m0.fit(X0, Y0)
	m1.fit(X1, Y1)
	output = open('models.pkl', 'wb')
	pickle.dump((m0,m1), output)
	output.close()

def demo():
	truths = pd.read_csv('truth.csv', header=None, index_col=0, prefix='Y')
	features = pd.read_csv("data.csv", header=None, index_col=0, prefix='X')
	data = features.join(truths)

	#import ipdb; ipdb.set_trace()

	data['X5'] = np.sqrt((data.X2 - data.X3)**2)
	data['X6'] = 100*data.X2 * data.X3
	#data['X6'] = data.X1*(data.X2 - data.X3)
	data['X7'] = data.X2 - data.X3
	#data['X8'] = data.X4**2
	#data['X8'] = data.X1*data.X2
	#data['X9'] = data.X1*data.X3

	data0 = data[data.X1>0]
	data1 = data[data.X1<0]
	#labeled0 = data0[~data0.Y1.isnull() & (data0.Y1 != 'Unknown')]
	#labeled1 = data1[~data1.Y1.isnull() & (data1.Y1 != 'Unknown')]
	labeled0 = data0[~data0.Y1.isnull() ]
	labeled1 = data1[~data1.Y1.isnull() ]

	labeled0[['X7','X6', 'X5','X2','X3','X4', 'Y1']].boxplot(by='Y1'); plt.show()
	labeled1[['X7','X6', 'X5','X2','X3','X4', 'Y1']].boxplot(by='Y1'); plt.show()
	#import ipdb; ipdb.set_trace()
	#X0 = labeled0.select_dtypes(include=['floating']).values
	X0 = labeled0[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
	Y0 = labeled0.Y1.values
	#X1 = labeled1.select_dtypes(include=['floating']).values
	X1 = labeled1[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
	Y1 = labeled1.Y1.values

	#m = DecisionTreeClassifier(max_depth=9)
	#m = svm.SVC()
	#m = LogisticRegression()
	m0 = OneVsRestClassifier(GaussianNB())
	m1 = OneVsRestClassifier(GaussianNB())
	#m = RandomForestClassifier(n_estimators=10)
	scores0 = cross_validation.cross_val_score(m0, X0, Y0, cv=3, n_jobs=-1)
	scores1 = cross_validation.cross_val_score(m1, X1, Y1, cv=3, n_jobs=-1)
	print scores0
	print scores1
	#import ipdb; ipdb.set_trace()

	m0.fit(X0, Y0)
	m1.fit(X1, Y1)

	#tree.export_graphviz(m,out_file='tree.dot') 
	#testX0 = data0.select_dtypes(include=['floating']).values
	#testX1 = data1.select_dtypes(include=['floating']).values
	testX0 = data0[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
	testX1 = data1[['X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values

	data0['testY'] = m0.predict(testX0)
	data0['lik'] = np.max( m0.predict_proba(testX0), axis=1 )
	data1['testY'] = m1.predict(testX1)
	data1['lik'] = np.max( m1.predict_proba(testX1), axis=1 )
	#labeled['testY'] = m.predict(X)
	import ipdb; ipdb.set_trace()
	for k, v in data0.groupby('testY'):
		dname = "./res/"+k
		if not os.path.exists(dname):
			os.makedirs(dname)
		for idx in v.index:
			#import ipdb; ipdb.set_trace()
			lik = int(100*v.loc[idx]['lik'])
			shutil.copy2("figs/%s.png"%(idx), dname+'/%s_%d.png'%(idx, lik))
	for k, v in data1.groupby('testY'):
		dname = "./res/"+k
		if not os.path.exists(dname):
			os.makedirs(dname)
		for idx in v.index:
			lik = int(100*v.loc[idx]['lik'])
			shutil.copy2("figs/%s.png"%(idx), dname+'/%s_%d.png'%(idx, lik))
	#import ipdb; ipdb.set_trace()
if __name__ == '__main__':
	trainModel()