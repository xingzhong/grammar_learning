#!/usr/bin/env python 
import numpy as np
np.set_printoptions(precision=3)
from pcfg import sample, Dist

def ntIndex(G, nt):
	for r, (nta, ntb, ntc, p) in enumerate(G, start=1):
		if nta == nt:
			yield r

def cyk (S, G):
	# S is sample , G is grammar
	n = S.shape[0]
	N = len(G)
	pi = -np.inf * np.ones((n+1, n+1, N+1))
	bp = np.empty_like(pi, dtype=set)
	for i, s in enumerate(S, start=1):
		for j, (nt, t, none, p) in enumerate(G, start=1):
			if none is None:
				pi[i, i, j] = np.log(p) + t.logPdf(s)
				bp[i, i, j] = ((i,i,j,nt), (-1, -1, -1, t), None)
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
									argmax = ( 
										(i,j, g, nta), 
										(i, s, bb, ntb), 
										(s+1, j, cc, ntc))
					pi[i, j, g] = maxx
					bp[i, j, g] = argmax

	build(bp, bp[1, n, N])
	
	return pi[1, n, N]

def build(bp, root):
	a, b, c = root
	if c:
		(i,j, g, nta) = a
		(i, s1, bb, ntb) = b 
		(s2, j, cc, ntc) = c
		print "%s[%s:%s] -> %s[%s:%s] %s[%s:%s]"%(nta, i, j, ntb, i, s1, ntc, s2, j)
		build(bp, bp[i, s1, bb])
		build(bp, bp[s2, j, cc])
	else:
		(i,j, g, nta) = a
		(i, s1, bb, ntb) = b
		print "%s[%s] -> %s[%s]"%(nta, i, ntb, i)

if __name__ == '__main__':
	_, S = sample()
	print S
	S = np.random.normal( 0, 1, size=(15,1))
	print S
	A = Dist(np.array([0.0]), np.array([0.1]))
	B = Dist(np.array([5.0]), np.array([0.1]))
	C = Dist(np.array([-5.0]), np.array([0.1]))

	G = [ ('NT_1', A, None, 1.0), ('NT_2', B, None, 1.0), ("NT_3", C, None, 1.0), 
			("NT_4", "NT_1", "NT_1", 0.2), ("NT_4", "NT_4", "NT_1", 0.8),
			("NT_5", "NT_2", "NT_2", 0.2), ("NT_5", "NT_5", "NT_2", 0.8),
			("NT_6", "NT_3", "NT_3", 0.2), ("NT_6", "NT_6", "NT_3", 0.8),
			("NT_7", "NT_5", "NT_4", 1.0), ("NT_S", "NT_7", "NT_6", 1.0) ]

	print cyk(S, G)
	
	


