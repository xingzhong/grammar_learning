from qcyk import qcyk
import tempfile
from terminal import terminal
import matplotlib.pyplot as plt

CS = {'RC': "#009933", 'BUR':"#009933", "BUT":"#009933",
		"IHS":"#009933", "BBOT":"#009933", "FW":"#009933", 
		"BER":"#CC3300", "BET":"#CC3300", "BTOP":"#CC3300", 
		"FC":"#CC3300", "HS":"#CC3300", "RW":"#CC3300"}

if __name__ == '__main__':
	likCache = tempfile.NamedTemporaryFile()
	data = terminal('aapl', likCache.name, alpha=-3.0)
	parser = qcyk()
	parser.initGrammar('test.gr')
	parser.initTerminal(likCache.name)
	lik, tree = parser.parse()
	likCache.close()

	print parser.pretty_print_tree(tree)
	print lik
	

	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(111)
	kps = data[data.pivots!=0].Close
	idx = kps.index
	kps.plot(style='--o', ax=ax)
	for terminal, xmin, xmax in parser.leafs(tree):
		center = idx[(xmin+xmax)/2]
		high = data[idx[xmin]:idx[xmax]].Close.max()
		low = data[idx[xmin]:idx[xmax]].Close.min()
		bottom, top = ax.get_ylim()
		llow = (low-bottom)/float(top-bottom)
		hhigh = (high-bottom)/float(top-bottom)
		if terminal != 'null':
			ax.axvspan(idx[xmin], idx[xmax], 
				ymin=llow, ymax=hhigh, alpha=0.3, color=CS[terminal])
			ax.annotate("%s"%(terminal), (center, high))
	plt.show()
	#import ipdb; ipdb.set_trace()