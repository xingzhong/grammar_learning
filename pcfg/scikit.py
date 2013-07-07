from sklearn import pcfg
import numpy as np

def grammar ():
	r2 = pcfg.Rule()
	UP = pcfg.NT('UP')
	ST = pcfg.NT('ST')
	DO = pcfg.NT('DO')
	u = pcfg.T('u', [1.0, 1.0, 1.0], 0.4*np.identity(3))
	s = pcfg.T('s', [0.0, 0.0, 0.0], 0.4*np.identity(3))
	d = pcfg.T('d', [-1.0, -1.0, -1.0], 0.4*np.identity(3))
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

g = grammar()
s = data ()
model = pcfg.BasePCFG(grammar=g)


print g
print s
lik, t = model.decode(s)
print model

print t

t.draw()
