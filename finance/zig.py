from scipy.stats import norm
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from zigzag import peak_valley_pivots, max_drawdown, compute_segment_returns, pivots_to_modes
from sklearn.mixture import GMM
import numpy as np
import csv
from parsing import parse
from sklearn.pcfg import Terminal

def preProcess(zig):
	f1 = pd.rolling_apply(zig, 5, lambda x: 100*(x[2]-x[0])/x[0], center = True, min_periods=5)
	f2 = pd.rolling_apply(zig, 5, lambda x: 100*(x[4]-x[2])/x[2], center = True, min_periods=5)
	f3 = pd.rolling_apply(zig, 5, lambda x: 100*(x[4]-x[0])/x[0], center = True, min_periods=5)
	f4 = pd.rolling_apply(zig, 5, lambda x: 100*(x[3]-x[1])/x[1], center = True, min_periods=5)
	f5 = pd.rolling_apply(zig, 5, lambda x: 100*(x[1]-x[0])/x[0], center = True, min_periods=5)
	features = pd.DataFrame({'f1':f1, 'f2':f2, "f3":f3, "f4":f4, 'f5':f5})
	return features.dropna()

def fetch(ticker='spy', start=datetime.datetime(2008, 1, 1),
		 end=datetime.datetime(2014, 1, 1)):
	return web.DataReader(ticker, 'google', start, end)

def zig(data):
	pivots = peak_valley_pivots(data.Close, 0.02, -0.02)
	return pivots

def f2p(features):
	# features => prob liks table
	X = features.values
	hs = GMM(1)
	hs.means_ = np.atleast_2d([10,-10,0,0, -10])
	hs.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	ihs = GMM(1)
	ihs.means_ = np.atleast_2d([-10,10,0,0, 10])
	ihs.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	btop = GMM(1)
	btop.means_ = np.atleast_2d([10,10,20,-10, -10])
	btop.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	bbot = GMM(1)
	bbot.means_ = np.atleast_2d([-10,-10,-20,10, 10])
	bbot.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	ttop = GMM(1)
	ttop.means_ = np.atleast_2d([-10,-10,-20,10, -10])
	ttop.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	tbot = GMM(1)
	tbot.means_ = np.atleast_2d([10,10,20,-10, 10])
	tbot.covars_ = np.array([[  1  , 1,  1,   1, 1]])

	liks = pd.DataFrame(index=features.index)
	liks['hs'] = hs.score(X)
	liks['ihs'] = ihs.score(X)
	liks['btop'] = btop.score(X)
	liks['bbot'] = bbot.score(X)
	liks['ttop'] = ttop.score(X)
	liks['tbot'] = tbot.score(X)
	return liks

def buildCSV(liks):
    with open('sample2.csv', 'wb') as f:
    	writer = csv.writer(f)
    	for idx, (row_index, row) in enumerate(liks.iterrows()):
        	for t, p in row.iteritems():
        		writer.writerow([t, idx, idx+4, p])

def main():

	data = fetch(ticker='spy', start=datetime.datetime(2010, 1, 1),
		 end=datetime.datetime(2012, 10, 1))
	kps = data[zig(data)!=0].Close
	features = preProcess(kps)
	ptables  = f2p(features)
	ptables.to_csv('test.csv')
	buildCSV(ptables)
	res = parse(ptables)
	idx = kps.index
	#

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(211)
	
	#data.Close.plot(style='.', ax=ax)
	kps.plot(style='--o', ax=ax)
	#
	bx = fig.add_subplot(212, sharex=ax)
	#features.plot(style='--o', ax=bx)
	ptables.plot(style='o', ax=bx)
	#import ipdb; ipdb.set_trace()
	for terminal, (xmin, xmax), _, _ in res:
		
		center = idx[(xmin+xmax)/2]
		high = data[idx[xmin]:idx[xmax]].Close.max()
		low = data[idx[xmin]:idx[xmax]].Close.min()
		
		bottom, top = ax.get_ylim()
		llow = (low-bottom)/float(top-bottom)
		hhigh = (high-bottom)/float(top-bottom)
		ax.axvspan(idx[xmin], idx[xmax], ymin=llow, ymax=hhigh, alpha=0.3)
		ax.annotate(terminal, (center, high))
		if terminal in ( Terminal('hs', None, None), Terminal('ihs', None, None)) :
			ax.plot([idx[xmin+1], idx[xmax-1]], [kps[xmin+1], kps[xmax-1]], c='r' )
			ax.plot([idx[xmin], idx[xmax]], [kps[xmin], kps[xmax]], c='r' )
	ax.set_xlim([idx[0]-timedelta(weeks=1), idx[-1]])
	plt.show()

if __name__ == '__main__':
	main()

