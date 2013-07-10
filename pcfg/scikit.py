from sklearn import pcfg
import numpy as np
from pprint import pprint 

def grammar ():
	r2 = pcfg.Rule()
	UP = pcfg.NT('UP')
	ST = pcfg.NT('ST')
	DO = pcfg.NT('DO')
	u = pcfg.T('u', [10.0, 10.0], 0.2*np.identity(2))
	s = pcfg.T('s', [0.0, 0.0], 0.2*np.identity(2))
	d = pcfg.T('d', [-10.0, -10.0], 0.2*np.identity(2))
	empty = pcfg.Empty(None)
	r2[(UP, u, UP, u)] = 0.45
	r2[(UP, u, ST, u)] = 0.45
	r2[(UP, empty)] = 0.1
	r2[(ST, s, UP)] = 0.33
	r2[(ST, s, ST)] = 0.33
	r2[(ST, s, DO)] = 0.33
	r2[(ST, empty)] = 0.01
	r2[(DO, ST, d)] = 0.4
	r2[(DO, DO, d)] = 0.4
	r2[(DO, empty)] = 0.2
	return r2

def data ():
	seq = np.array([
		[1.0,1.0,1.0], 
		[1.0,1.0,1.0], 
		[0.0,0.0,0.0],  
		[1.0,1.0,1.0],
		[-1.0, -1.0, -1.0],
		[1.0,1.0,1.0],
		])
	noise = 0.2*np.random.randn(*seq.shape)
	seq = seq + noise
	return seq

def sample_data(model):
	tree, sample = model.sample()
	return tree, sample

g = grammar()
#s = data ()
model = pcfg.BasePCFG(grammar=g)
#print g
t, s = sample_data(model)
print s.shape
t.draw()
lik, t = model.decode(s)
#print model
#print t
t.draw()

