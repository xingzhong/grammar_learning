from sklearn import pcfg
import numpy as np
from pprint import pprint 
import sys

sys.setrecursionlimit(9999)

def grammar_1 ():
	r2 = pcfg.Rule()
	TRI = pcfg.NT('TRI')
	TRID = pcfg.NT('TRID')
	ST = pcfg.NT('ST')
	u = pcfg.T('u', [1.0], 0.2*np.identity(1))
	s = pcfg.T('s', [0.0], 0.2*np.identity(1))
	d = pcfg.T('d', [-1.0], 0.2*np.identity(1))
	empty = pcfg.Empty(None)
	r2[(TRI, u, TRI, d)] = 0.2
	r2[(TRI, u, ST, d)] = 0.4
	r2[(TRI, TRI, TRI)] = 0.4
	r2[(ST, s, ST)] = 0.8
	r2[(ST, empty)] = 0.2
	return r2

def grammar_2 ():
	r2 = pcfg.Rule()
	LS = pcfg.NT('LS')
	ST = pcfg.NT('ST')
	u = pcfg.T('u', [1.0], 0.2*np.identity(1))
	s = pcfg.T('s', [0.0], 0.2*np.identity(1))
	d = pcfg.T('d', [-1.0], 0.2*np.identity(1))
	empty = pcfg.Empty(None)
	r2[(TRI, u, TRI, d)] = 0.8
	r2[(TRI, u, ST, d)] = 0.2
	r2[(ST, s, ST)] = 0.80
	r2[(ST, empty)] = 0.20
	return r2

def grammar_3 ():
	r2 = pcfg.Rule()
	S = pcfg.NT('S')
	UPT = pcfg.NT('UPT')
	DOT = pcfg.NT('DOT')
	STAY = pcfg.NT('STAY')
	up = pcfg.T('up', [1.0], 0.2*np.identity(1))
	stay = pcfg.T('stay', [0.0], 0.2*np.identity(1))
	down = pcfg.T('down', [-1.0], 0.2*np.identity(1))
	empty = pcfg.Empty(None)

	r2[(S, S, S)] = 0.6
	r2[(S, UPT, DOT)] = 0.4
	r2[(UPT, up, UPT, down)] = 0.7
	r2[(UPT, up, STAY, down)] = 0.3
	r2[(DOT, down, DOT, up)] = 0.7
	r2[(DOT, down, STAY, up)] = 0.3
	r2[(STAY, STAY, stay)] = 0.8
	r2[(STAY, empty)] = 0.2
	return r2

def data ():
	seq = np.array([
		[1.0], 
		[0.0], 
		[-1.0],  
		[1.0],
		[0.0],
		[0.0],
		[-1.0],
		])
	noise = 0.2*np.random.randn(*seq.shape)
	seq = seq + noise
	return seq

def sample_data(model):
	tree, sample = model.sample()
	pprint (sample)
	return tree, sample

s = data ()
model = pcfg.BasePCFG(grammar=grammar_3(), start=pcfg.NT('S'))

t, s = sample_data(model)
print s.shape

#t.draw()
#lik, t = model.decode(s)
#pprint( model.gamma )
#print t
#t.draw()
data = t.pos()


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
fig = plt.figure()
ax = fig.add_subplot(111)


df = pd.DataFrame(data, columns=['value', 'state'])
for n, g in df.groupby('state'):
	if n.type() == "T":
		print n
		print g['value']
		import pdb; pdb.set_trace()
		ax.plot(g, "o-")
ax.grid(True)
plt.show()
#import pdb; pdb.set_trace()
#from draw import draw
#draw(s, np.cumsum(s))
