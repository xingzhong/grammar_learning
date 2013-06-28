from sklearn import pcfg
import numpy as np

def grammar ():
	r2 = pcfg.Rule()
	A = pcfg.NT('A')
	B = pcfg.NT('B')
	C = pcfg.NT('C')
	a = pcfg.T('a', [0.0, 0.0], 0.4*np.identity(2))
	b = pcfg.T('b', [1.0, 1.0], 0.4*np.identity(2))
	c = pcfg.T('c', [-1.0, -1.0], 0.4*np.identity(2))
	empty = pcfg.Empty(None)
	r2[(A, a, A, a)] = 0.45
	r2[(A, a, B, a)] = 0.45
	r2[(A, empty)] = 0.1
	r2[(B, b, A)] = 0.33
	r2[(B, b, B)] = 0.33
	r2[(B, b, C)] = 0.33
	r2[(B, empty)] = 0.01
	r2[(C, B, c)] = 0.4
	r2[(C, C, c)] = 0.4
	r2[(C, empty)] = 0.2
	return r2

def data ():
	seq = np.array([[0.0,0.0], [1.0,1.0], [1.0,1.0], [-1.0, -1.0], [0.0, 0.0]])
	return seq

g = grammar()
s = data ()
model = pcfg.BasePCFG(grammar=g)

print g
print s
lik, t = model.decode(s)

print t

t.draw()