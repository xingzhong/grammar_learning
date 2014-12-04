from zig import fetch 
from zigzag import peak_valley_pivots
import numpy as np
import matplotlib.pyplot as plt
import csv



def trend(x, y):
	a = (y[1] - y[0])/(x[1] - x[0])
	b = y[1] - a * x[1]
	return a, b

def main(ticker, idx=0, alpha = 0.02):
	data =fetch(ticker)
	data['pivots'] = peak_valley_pivots(data.Close, alpha, -alpha)
	rd = data.reset_index()
	X = rd[rd.pivots!=0][['Close', 'pivots']].reset_index().values
	X = np.dstack( [X[i:i+5, :] for i in range( X.shape[0] -4)] )
	features = []
	for i in range(X.shape[-1]):
		x = X[:,:,i]
		x[:, 1] = x[:, 1]/x[0,1]
		a1, b1 = trend(x[::4, 0], x[::4, 1])
		a2, b2 = trend(x[1::2, 0], x[1::2, 1])
		xm = a1*x[2,0] + b1
		dy = xm - x[2,1]
		features.append([idx, x[0,2], a1, a2, dy])
		print idx, x[0,2], a1, a2, dy
		#import ipdb; ipdb.set_trace()
		raw = rd[int(x[0,0]): int(x[-1,0])]
		ret = raw.Close/raw.Close.iloc[0]
		ry1 = a1 * x[:,0] + b1 
		ry2 = a2 * x[:,0] + b2 
		fig = plt.figure(figsize=(6,3))
		ax = fig.add_subplot(111)
		ret.plot(style='g.', ax=ax)
		ax.plot(x[:,0], x[:,1], '--')
		ax.plot(x[:,0], ry1, 'r-', lw=2)
		ax.plot(x[:,0], ry2, 'b-', lw=2)
		ax.plot(x[::2, 0], x[::2, 1], 'ro')
		ax.plot(x[1::2, 0], x[1::2, 1], 'bo')
		ax.axhline(y=1.0, lw=1, c='k')
		ax.annotate("%.3f%%"%(100*a1), (x[1,0], a1*x[1,0]+b1), color='r')
		ax.annotate("%.3f%%"%(100*a2), (x[3,0], a2*x[3,0]+b2), color='b')
		ax.annotate("%.3f%%"%(100*dy), (x[2,0], x[2,1]), color='k')
		ax.set_title("#%s %s, %.3f%%, %.3f%%, %.3f%%"%(idx, x[0,2], 100*a1, 100*a2, 100*dy))
		ax.set_xlim([x[0,0]-5, x[-1,0]+5])
		ax.set_ylim([x[:,1].min()-0.03, x[:,1].max()+0.03 ])
		plt.savefig("figs/%s.png"%(idx))
		idx += 1
	with open('data.csv', 'a') as fs:
		writer = csv.writer(fs)
		writer.writerows(features)
	return idx
		#plt.show()
		#break

if __name__ == '__main__':
	idx = 0
	for ticker in ['spy', 'aapl', 'gld', 'iwm', 'bac', 'gs']:
		for alpha in [0.02, 0.04, 0.06]:
			idx = main(ticker, idx=idx, alpha=alpha)
