# quick CYK parsing 
# optimum alignment sequence through dynamic programming 
# input:
# 	1. rule sets
#	2. terminal liklihood 
# output:
#	optimum decoding alignment and structural tree
import math

class qcyk(object):
	MIN = -1e11
	def __init__(self):
		self.gamma = {}
		self.tau = {}
		self.rules = {}
		self.rules_lhs = {}
		self.length = 0
	def initTerminal(self, fin):
		self.length = 0
		with open(fin, 'r') as f:
			for line in f:
				start, end, terminal, logP = line.split(',')
				self.gamma[(int(start), int(end), terminal.strip())] = float(logP)
				self.tau[(int(start), int(end), terminal.strip())] = (None, None, -1)
				if int(end) > self.length:
					self.length = int(end)
		#self.length += 1

	def initRules(self, fin):
		with open(fin, 'r') as f:
			for line in f:
				a, b, c, p = line.split(',')
				a, b, c, p = a.strip(), b.strip(), c.strip(), math.log(float(p))
				if len(c) < 1: c = None
				self.rules[(a,b,c)] = p
				if a in self.rules_lhs:
					self.rules_lhs[a].append( (b, c, p) )
				else:
					self.rules_lhs[a] = [ (b,c,p) ]

	def parse(self):
		lik = self.getGamma(0, self.length, 'S')
		if lik > qcyk.MIN:
			return lik, self.tree(0, self.length, 'S')

	def tree(self, start, end, nt):
		root = (nt, start, end, [])
		y, z, k = self.tau[(start, end, nt)]
		if y:
			root[-1].append( self.tree(start, k, y) )
		if z:
			root[-1].append( self.tree(k+1, end, z) )
		return root	

	def getGamma(self, i, j, v):
		# v[i:k:j] -> y z 
		
		if (i, j, v) in self.gamma:
			return self.gamma[(i, j, v)]
		elif (i > j) or (v not in self.rules_lhs):
			return qcyk.MIN
		else:
			y0, z0, k0, lik0 = None,None,-1,-1e10
			print self.rules_lhs[v]
			for (y, z, logP) in self.rules_lhs[v]:
				print i, j, v, y, z
				if z and (i!=j):
					for k in range(i, j):
						lik = self.getGamma(i, k, y) + self.getGamma(k+1, j, z) + logP
						if lik > lik0:
							y0, z0, k0, lik0 = y,z,k,lik
				elif not z:
					#import ipdb; ipdb.set_trace()
					lik = self.getGamma(i, j, y) + logP
					if lik > lik0:
						y0, z0, k0, lik0 = y,None,j,lik


			if k0 > -1:
				self.gamma[(i, j, v)] = lik0
				self.tau[(i, j, v)] = (y0, z0, k0)
				
			else:
				self.gamma[(i, j, v)] = qcyk.MIN
				self.tau[(i, j, v)] = (y0, z0, k0)

			return self.gamma[(i, j, v)]
			raise AssertionError("No rules for nonterminal %s"%v)

if __name__ == '__main__':
	parser = qcyk()
	parser.initTerminal("qcyk_cal_liks.csv")
	parser.initRules('qcyk_cal_rules.csv')
	print parser.parse()
	
	