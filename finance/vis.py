from qcyk import qcyk
import tempfile
import cStringIO
from terminal import terminal
import matplotlib.pyplot as plt
import os


CS = {'RC': "#009933", 'BUR':"#009933", "BUT":"#009933",
		"IHS":"#009933", "BBOT":"#009933", "FW":"#009933",
		"BER":"#CC3300", "BET":"#CC3300", "BTOP":"#CC3300",
		"FC":"#CC3300", "HS":"#CC3300", "RW":"#CC3300"}

def vis(ticker):
	sio = cStringIO.StringIO()
	likCache = tempfile.NamedTemporaryFile()
	data = terminal(ticker, likCache.name, alpha=-3.0)
	parser = qcyk()
	parser.initGrammar(os.path.dirname(os.path.realpath(__file__))+'/test.gr')
	parser.initTerminal(likCache.name)
	lik, tree = parser.parse()
	likCache.close()

	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(111)
	kps = data[data.pivots!=0].Close
	idx = kps.index
	kps.plot(style='--o', ax=ax)
	for t, xmin, xmax in parser.leafs(tree):
		center = idx[(xmin+xmax)/2]
		high = data[idx[xmin]:idx[xmax]].Close.max()
		low = data[idx[xmin]:idx[xmax]].Close.min()
		bottom, top = ax.get_ylim()
		llow = (low-bottom)/float(top-bottom)
		hhigh = (high-bottom)/float(top-bottom)
		if t != 'null':
			ax.axvspan(idx[xmin], idx[xmax],
				ymin=llow, ymax=hhigh, alpha=0.3, color=CS[t])
			ax.annotate("%s"%(t), (center, high))

	plt.savefig(sio, format='png')
	sio.seek(0)
	return sio.getvalue()

if __name__ == '__main__':
	print vis('aapl')
