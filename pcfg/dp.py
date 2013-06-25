import numpy as np
import itertools
from pprint import pprint
from pdf import *

class NT(object):
	def __init__(self, symbol):
		self._symbol = symbol

	def type(self):
		return "NT"

	def __repr__(self):
		return "%s(%s)"%(self.type(), self._symbol)

	def __eq__(self, other):
		return self._symbol == other._symbol

	def __hash__(self):
		return hash(self._symbol)

class T(NT):
	def __init__(self, symbol, dist=None):
		self._symbol = symbol
		self._dist = dist

	def dist(self):
		return self._dist

	def loadDist(self, dist):
		self._dist = dist

	def type(self):
		return "T"


class Empty(NT):
	def type(self):
		return "Empty"

class Rule(dict):
	def get(self, lhs=None, rhs=None):
		if lhs and not rhs:
			return {k:v for (k,v) in self.iteritems()\
								if k[0] == lhs}
		if lhs and rhs:
			return {k:v for (k,v) in self.iteritems()\
								if k[0] == lhs and rhs in k}

	def sample(self):
		a = T('a', Dist( 0, .4))
		b = T('b', Dist( 1, .4))
		c = T('c', Dist(-1, .4))
		A = NT('A')
		B = NT('B')
		C = NT('C')
		empty = Empty(None)
		self[(A, a, A, a)] = 0.45
		self[(A, a, B, a)] = 0.45
		self[(A, empty)] = 0.1
		self[(B, b, A)] = 0.33
		self[(B, b, B)] = 0.33
		self[(B, b, C)] = 0.33
		self[(B, empty)] = 0.01
		self[(C, B, c)] = 0.4
		self[(C, C, c)] = 0.4
		self[(C, empty)] = 0.2

class Tree(object):
	def __init__(self, node):
		self.node = node 
		self.children = []

	def add(self, node):
		self.children.append(node)

	def __repr__(self):
		
		return "[%s  %s]"%(str(self.node[0]), 
			" ".join(map(str, self.children)))

class Gamma(dict) :
	# gamma((i,j,m)):
	# 	key is three tuple
	# 	i : start index 
	# 	j	: end index
	#		m : production lhs
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self._chart = {}

	def __repr__(self):
		s = " ".join(map(str, self._seq))
		s += '\n---chart---\n'
		for row in range(self._maxL+1):
			for nt in [NT('A'), NT('B'), NT('C')]:
				for col in range(self._maxL):
					i = col + 1
					j = col + self._maxL - row 
					if j <= self._maxL:
						s += str((i, j, nt))
						if self.has_key((i,j,nt)):
							s += "%8.4f  "%self[(i, j, nt)]
						else:
							s += " "*10
				s += '\n'
		s+='----end----\n'
		s+="---tree----\n"
		s+=str( self.tree() )
		s+='\n----end----\n'
		return s
	
	def tree(self):
		start = NT("A")
		return self._tree ( (1, self._maxL, start) )


	def _tree(self, key):
		start = key[0]
		end = key[1]
		value = self._chart[key]
		node = Tree( (key[2], (start, end)) )
		
		if isinstance(value, Empty):
			node.add( value )
		else:
			if value[0].type() == "T":
				sn = Tree((value[0], start, start))
				sn.add(self._seq[start-1])
				node.add( sn )
				start = start + 1

			en = None
			if value[-1].type() == "T":
				en = Tree((value[-1], end, end))
				en.add(self._seq[end-1])
				end = end - 1

			for xx in value:
				if xx.type() == "NT":
					node.add( self._tree( (start, end, xx) ))

			if en:
				node.add( en )

		
		return node

	def loadRule(self, r):
		self._rule = r
	
	def loadSeq(self, s):
		self._seq = s
		self._maxL = len(s)
	
	def rule(self):
		return self._rule
	
	def take(self, i, j, m):
		#print "(%s, %s, %s)"%(i,j,m)
		if i-1 == j and i<=self._maxL and j>0:
			r = self._rule.get(lhs=m, rhs=Empty(None))
			#print 'log P(%s->empty)'%(m)
			self._chart[(i,j,m)] = Empty(None)
			self[(i,j,m)] = np.log(r.values()[0])
			return np.log(r.values()[0])
		elif i > j:
			self[(i,j,m)] = np.log(0.0)
			return np.log(0.0)
		elif i > self._maxL:
			self[(i,j,m)] = np.log(0.0)
			return np.log(0.0)
		elif self.has_key((i, j, m)):
			return self[(i,j,m)]
		else:
			temp = []
			for r in self._rule.get(lhs = m):
				# iterate into each production(lhs=m)
				rhs = r[1:]
				#print m, r
				w = filter(lambda x: x.type()=='NT', rhs)
				if len(w)>0:
					# not none rhs
					w = w[0]
					#print w
					if len(filter(lambda x:x.type()=='T', rhs))>1:
						# double sides
						
						_lik = rhs[0].dist().logpdf(self._seq[i-1])
						_lik = _lik + rhs[2].dist().logpdf(self._seq[j-1])
						
						#print '\tgamma(%s, %s, %s) + log P(%s->%s %s %s)'\
						#		%(i+1, j-1, w, m, self._seq[i-1], w, self._seq[j-1])
						lik = self.take(i+1,j-1, w) + _lik + np.log(self._rule[r])
						
						print '(%s,%s,%s) = gamma(%s, %s, %s) + log P(%s->%s %s %s) = %s'\
								%(i, j, m, i+1, j-1, w, m, self._seq[i-1], w, self._seq[j-1], lik)
						bp = (i, j, r)
						temp.append((lik, bp))
					
					elif rhs[0].type() == "T":
						# left				
						_lik = rhs[0].dist().logpdf(self._seq[i-1])		
						
						#print '\tgamma(%s, %s, %s) + log P(%s->%s %s)'\
						#		%(i+1, j, w, m, self._seq[i-1], w)
						lik = self.take(i+1,j, w) + _lik + np.log(self._rule[r])
						
						print '(%s,%s,%s) = gamma(%s, %s, %s) + log P(%s->%s %s) = %s'\
								%(i,j,m,i+1, j, w, m, self._seq[i-1], w, lik)
						bp = (i, j, r)
						temp.append((lik, bp))

					elif rhs[0].type() == "NT":
						# right	
						_lik = rhs[1].dist().logpdf(self._seq[j-1])	

						
						#print '\tgamma(%s, %s, %s) + log P(%s-> %s %s)'\
						#		%(i, j-1, w, m, w, self._seq[j-1])
						lik = self.take(i,j-1, w) + _lik + np.log(self._rule[r])
						
						print '(%s,%s,%s) = gamma(%s, %s, %s) + log P(%s-> %s %s) = %s'\
								%(i,j,m,i, j-1, w, m, w, self._seq[j-1], lik)
						bp = (i, j, r)
						temp.append((lik, bp))

					else:
						lik = 0.0

			if len(temp)>0:
				self[(i,j,m)], bp = max(temp)
				#self._chart.append(bp)
				self._chart[(bp[0], bp[1], bp[2][0])] = bp[2][1:]
				return self[(i,j,m)]

			else:
				self[(i,j,m)] = np.log(0.0)
				return np.log(0.0)

if __name__ == '__main__':
	t1 = 'a b b c a'
	t2 = 'a b b b a b a c a'
	t3 = [0.0, 1.0, 1.0, -1.0, 0.0]
	t4 = t3 + Dist(0,.2).rvs(5)
	t5 = Dist(0,1).rvs(10)
	g = Gamma()
	r = Rule()
	r.sample()
	#s = map(T, t3.split())
	s = t4
	print s
	g.loadRule(r)
	g.loadSeq(s)

	pprint(g.rule())

	
	g.take(1,len(s),NT('A'))
	print g
	g.tree()

	#import pdb; pdb.set_trace()
	

	
