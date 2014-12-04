import pickle
from zig import fetch 
from zigzag import peak_valley_pivots
import numpy as np
import pandas as pd
from parsing import parse 
import matplotlib.pyplot as plt
import datetime 

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
	
def likTable(F,m0,m1):
	res = []
	for f in F:
		if f[0]>0:
			res.append( dict( zip(m0.classes_.tolist(), m0.predict_proba(f[1:]).tolist()[0] ) ) )
		else:
			res.append( dict( zip(m1.classes_.tolist(), m1.predict_proba(f[1:]).tolist()[0] ) ) )
	return np.log(pd.DataFrame(res).fillna(0))


def parseData(ticker):
	modelsFile = open('models.pkl', 'rb')
	m0, m1 = pickle.load(modelsFile)
	F, data = dataXY(ticker)
	ptable = likTable(F, m0, m1)
	res = parse(ptable)
	kps = data[data.pivots!=0].Close
	idx = kps.index

	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(211)
	bx = fig.add_subplot(212)
	kps.plot(style='--o', ax=ax)
	ptable.plot(style='o', ax=bx)
	#import ipdb; ipdb.set_trace()
	for terminal, (xmin, xmax), _, logLik in res:
		center = idx[(xmin+xmax)/2]
		high = data[idx[xmin]:idx[xmax]].Close.max()
		low = data[idx[xmin]:idx[xmax]].Close.min()
		
		bottom, top = ax.get_ylim()
		llow = (low-bottom)/float(top-bottom)
		hhigh = (high-bottom)/float(top-bottom)
		ax.axvspan(idx[xmin], idx[xmax], ymin=llow, ymax=hhigh, alpha=0.3)
		ax.annotate("%s"%(terminal), (center, high))
		#ax.annotate("%s[%.3f]"%(terminal,logLik), (center, high))
	bx.legend(loc=4)
	plt.savefig("parsed/%s.png"%(ticker))
	#plt.show()
if __name__ == '__main__':
	for t in ['spy', 'iwm', 'gld', 'gs', 'aapl', 'goog', 'xom', 'amzn', 'bby', 'tsla']:
		parseData(t)