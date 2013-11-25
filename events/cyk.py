#!/usr/bin/env python 
import numpy as np
np.set_printoptions(precision=3)
from pcfg import sample, Dist

def ntIndex(G, nt):
	for r, (nta, ntb, ntc, p) in enumerate(G, start=1):
		if nta == nt:
			yield r

def cyk ():
	N = 10
	S = np.random.choice( ['A', "B", "C"], size=N)
	S = ['A', 'B', 'B', 'C']
	N = 4
	G = [ ('NT_A', 'A', None), ('NT_B', "B", None), ("NT_C", "C", None), 
			("NT_AB", "NT_A", "NT_B"), ("NT_BC", "NT_B", "NT_C"), ("NT_S", "NT_AB", "NT_BC")]
	R = len(G)
	P = np.zeros((N+1, N+1, R+1))
	print " ".join(S)
	for i, s in enumerate(S):
		for j, (nt, t, none) in enumerate(G):
			if s == t and none is None:
				P[i+1, 1, j+1] = 1.0
				print "%s[%s] ->%s"%(nt, i+1, t)
	
	for ii in range(2, N+1):
		for jj in range(1, N-ii+2):
			for kk in range(1, ii):
				for rr, (nta, ntb, ntc) in enumerate(G):
					if ntc:
						bb = ntIndex(G, ntb)
						cc = ntIndex(G, ntc)
						aa = ntIndex(G, nta)
						
						if P[jj, kk, bb] != 0 and P[jj+kk, ii-kk, cc] != 0:
							print "%s[%s:%s] -> %s %s"%(nta, jj, jj+ii-1, ntb, ntc)
							P[jj, ii, aa] = 1.0
	print P[1, N, R]
	#print P
if __name__ == '__main__':
	_, S = sample()
	A = Dist(np.array([0.0]), np.array([0.1]))
	B = Dist(np.array([5.0]), np.array([0.1]))
	C = Dist(np.array([-5.0]), np.array([0.1]))

	G = [ ('NT_1', A, None, 1.0), ('NT_2', B, None, 1.0), ("NT_3", C, None, 1.0), 
			("NT_4", "NT_1", "NT_1", 0.2), ("NT_4", "NT_4", "NT_1", 0.8),
			("NT_5", "NT_2", "NT_2", 0.2), ("NT_5", "NT_5", "NT_2", 0.8),
			("NT_6", "NT_3", "NT_3", 0.2), ("NT_6", "NT_6", "NT_3", 0.8),
			("NT_7", "NT_5", "NT_4", 1.0), ("NT_S", "NT_7", "NT_6", 1.0) ]
	n = S.shape[0]
	N = len(G)
	pi = -np.inf * np.ones((n+1, n+1, N+1))
	bp = np.empty_like(pi, dtype=set)
	for i, s in enumerate(S, start=1):
		for j, (nt, t, none, p) in enumerate(G, start=1):
			if none is None:
				pi[i, i, j] = np.log(p) + t.logPdf(s)
				bp[i, i, j] = (nt, i, t, j)
				#print "%s pi[%s, %s, %s] = %s"%(s, i,i,nt,pi[i, i, j])
	
	for l in range(1, n):
		for i in range(1, n-l+1):
			j = i+l
			for g, (nta, ntb, ntc, p) in enumerate(G, start=1):
				if ntc:
					
					maxx, argmax = -np.inf, None
					for bb in ntIndex(G, ntb):
						for cc in ntIndex(G, ntc):
							for s in range(i, j):					
								prob = pi[i, s, bb] + pi[s+1, j, cc] + np.log(p)
								
								if prob > maxx:
									maxx = prob
									argmax = (nta, ntb, ntc)
					
					pi[i, j, g] = maxx
					bp[i, j, g] = argmax
					print i,j,g, pi[i, j, g], argmax
	
	print pi[1, n, N]
	print S.T

	import pdb; pdb.set_trace()

