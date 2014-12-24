import pickle
import pandas.io.data as web
from zigzag import peak_valley_pivots
import numpy as np
import pandas as pd
import datetime
import os

def fetch(ticker='spy', start=datetime.datetime(2008, 1, 1),
		 end=datetime.datetime.now()):
	return web.DataReader(ticker, 'google', start, end)

def trend(x, y):
	a = (y[1] - y[0])/(x[1] - x[0])
	b = y[1] - a * x[1]
	return a, b

def genFeature(m):

	a1, b1 = trend(m[::4, 0], m[::4, 1])
	a2, b2 = trend(m[1::2, 0], m[1::2, 1])
	xm = a1*m[2,0] + b1
	dy = xm - m[2,1]
	return m[0,2], a1, a2, dy, np.sqrt((a1-a2)**2), a1*a2, a1-a2

def dataXY(ticker, alpha=0.02):
	data =fetch(ticker, start=datetime.datetime(2012, 1, 1))
	data['pivots'] = peak_valley_pivots(data.Close, alpha, -alpha)
	rd = data.reset_index()
	X = rd[rd.pivots!=0][['Close', 'pivots']].reset_index().values
	X = np.dstack( [X[i:i+5, :] for i in range( X.shape[0] -4)] )
	X[:,1,:] = X[:,1,:]/X[0,1,:] # covert to return axis
	F = map(lambda i: genFeature(X[..., i]), range(X.shape[-1]))
	F = np.array(F)
	return F, data

def likTable(F,m0,m1, fn, alpha):
	with open(fn, "wb") as fileN:
		for idx, f in enumerate(F):
			if f[0]>0:
				d = zip(m0.classes_.tolist(), m0.predict_proba(f[1:]).tolist()[0] )
			else:
				d = zip(m1.classes_.tolist(), m1.predict_proba(f[1:]).tolist()[0] )
			for k, v in d:
				fileN.write("%d,%d,%s,%f\n"%(idx, idx+4, k, np.log(v)))
		for i in range(len(F)+4):
			fileN.write("%d,%d,%s,%f\n"%(i, i, 'null', alpha))

def terminal(ticker='spy', fn='test.csv', alpha=-2.0):
    model = os.path.dirname(os.path.realpath(__file__)) + '/models.pkl'
    modelsFile = open(model, 'rb')
    m0, m1 = pickle.load(modelsFile)
    F, data = dataXY(ticker)
    likTable(F, m0, m1, fn, alpha)
    return data


if __name__ == '__main__':
	terminal()
