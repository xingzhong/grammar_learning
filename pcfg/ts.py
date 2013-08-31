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

def grammar_4():
    r = pcfg.Rule()
    S = pcfg.NT('S')
    a = pcfg.T('a', [1.0,1.0], 0.2*np.identity(2))
    c = pcfg.T('c', [1.0,-1.0], 0.2*np.identity(2))
    empty = pcfg.Empty(None)
    r[(S, S, S)] = 0.1
    r[(S, a, S, c)] = 0.8
    r[(S, empty)] = 0.1
    return r

def data ():
	seq = np.array([
		[1.0, 1.0], 
		[1.0, 0.0], 
		[1.0, 0.0],
		[1.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
		])
	noise = 0.2*np.random.randn(*seq.shape)
	seq = seq + noise
	return seq

def sample_data(model):
	tree, sample = model.sample()
	pprint (sample)
	return tree, sample

s = data ()
model = pcfg.BasePCFG(grammar=grammar_4(), start=pcfg.NT('S'))

tree, sample = model.sample()
print sample

lik, t, tag = model.decode(s)

print lik

print model.score(s)
