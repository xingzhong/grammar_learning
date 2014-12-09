import pandas as pd
import matplotlib.pyplot as plt
from sklearn import hmm
import numpy as np
from cyk import cyk
from pcfg import  Dist

class T(object):
	def __init__(self, id, value):
		self._id = id
		self._value = value

	def value(self):
		return self._value

def markovLearning(s):
	model = hmm.GaussianHMM(3, 'diag')
	model.fit([s])
	print model.means_
	#print model.covars_
	print model.transmat_
	return model

def grammar():
	G = [ ('NT_0', ['NT_0', T(0, -0.594)], 0.8), ('NT_0', [T(0, -0.594)], 0.2), 
			('NT_1', ['NT_1', T(1, -0.27)], 0.8), ('NT_1', [T(1, -0.27)], 0.2), 
			('NT_2', ['NT_2', T(2, -0.491),], 0.8), ('NT_2', [T(2, -0.491)], 0.2), 
			('NT_3', ['NT_3', T(3, 0.404)], 0.8), ('NT_3', [T(3, 0.404)], 0.2), 
			('NT_4', ['NT_4', T(4, -0.418)], 0.8), ('NT_4', [T(4, -0.418)], 0.2), 
			('NT_5', ['NT_1', 'NT_3', 'NT_1'], 1.0),
			('NT_6', ['NT_4', 'NT_5', 'NT_4'], 1.0),
			('NT_7', ['NT_4', 'NT_1'], 1.0),
			('NT_8', ['NT_7', 'NT_6'], 1.0),
			('NT_9', ['NT_2', 'NT_8', 'NT_2'], 1.0),
			('S', ['NT_0', 'NT_9', 'NT_0'], 1.0),
			]
	return G

def sampleG(g, root, container):
	if isinstance(root, T):
		container.append(root.value())
		return 
		#return root
	rules = filter(lambda x: x[0] == root, g)
	probs = map(lambda x: x[2], rules)
	idx = np.random.choice(len(rules), p=probs)
	rule = rules[idx]
	[ sampleG(g, rhs, container) for rhs in rule[1] ]

def ecg():
	table = pd.read_csv("sample.csv", sep=r'\s+', 
		index_col=0, names=['time','a','b'])
	idx = table.index[::230]
	for i,j in zip(idx, idx[1:]):
		yield table.a[i:j]

def draw(data, decode, model, title=None):
	figsize=(30,8)
	fig = plt.figure(figsize=figsize)
	for idx, dd in enumerate(data):
		if idx>17: 
			break
		ax = fig.add_subplot(6, 3, idx)
		try:
			ax.plot(dd.values)
			ax.set_title(title[idx])
		except:
			ax.plot(dd)
			continue
		try :
			ax.plot(decode[idx], color="red")
		except:
			
			continue
		for m in model.means_:
			ax.axhline(m[0], alpha=0.6, color='pink')
		
	plt.show()

def markov():
	data = [d for d in ecg()]
	raw =  np.array( [ d.values for d in data ] ) 
	sample = np.atleast_3d(raw)
	model = hmm.GaussianHMM(5, 'diag')
	model.fit(sample[:6])
	print model.means_
	lik = [ model.score(i) for i in sample ]
	decode = [ map(lambda x: model.means_[x][0], model.decode(i)[1]) for i in sample]
	#import pdb; pdb.set_trace()
	#simulate, _ = model.sample(231)
	#lik.append( model.score( simulate) )
	#data.append(simulate)
	#decode.append(model.decode(simulate))
	simulate, _ = model.sample(231)
	lik.append( model.score( simulate) )
	data.append(simulate)
	#decode.append(model.decode(simulate))
	

	g = grammar()
	simulate = []
	sampleG(g, 'S', simulate)
	#import pdb; pdb.set_trace()
	
	decode.append(simulate)
	#lik.append( model.score(simulate) )
	data.append(simulate)
	draw(data, decode, model, title=lik)
	
	return sample, model

def pcfg():
	data = [d for d in ecg()]
	raw =  np.array( [ d.values for d in data ] ) 
	sample = np.atleast_3d(raw)
	g = grammar()
	
	for seq in sample :
		#import pdb; pdb.set_trace()
		print cyk(seq, g)
if __name__ == '__main__':
	sample, model = markov()
	#g = grammar()
	#print sampleG(g, 'S')
	

	
