from pcfg import *
import pandas as pd

def ecg():
	table = pd.read_csv("sample.csv", sep=r'\s+', index_col=0, names=['time','a','b'])
	#import pdb; pdb.set_trace()
	g = EventGraph()
	for t, atom in enumerate (table.a[:230].values):
		if not atom is None:
			g.addEvent( Event(t, 0,  np.array([atom])))
	g.buildEdges(delta = 1)
	return g, table.a[:230].values

if __name__ == '__main__':
	figsize=(30,8)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(1, 1, 1)
	np.set_printoptions(precision=3)
	g, samples = ecg()
	#g, samples = sample()
	ax.plot(samples, 'o')
	gamma = -15
	grammar = Grammar(delta=-gamma)
	for i in range(20):
		print i, len(g.nodes())
		#VisGraph(g, ax, offset=i)
		#visTimeSeries(g, ax, offset=10*i, label=i)
		if len(g.nodes()) < 2:
			break
		prod = toProd(g, k=12, gamma=gamma)
		print prod
		grammar.addProd(prod)
		rewrite(g, prod, gamma=gamma)
		#import pdb; pdb.set_trace()
	
	pprint (grammar._nts)
	pprint (grammar._prods)
	
	for i, (key, nt) in enumerate(grammar._nts.iteritems()):
		if nt._mean.shape[0] == 1 :
			ax.axhline(y=nt._mean[0], ls='--', alpha=0.6, color='red')
			ax.text( i, nt._mean[0], key )
		
	
	plt.grid()
	#plt.legend()
	plt.show()
	#import pdb; pdb.set_trace()
	
	