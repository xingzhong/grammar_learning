import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
from collections import Counter

class T(tuple):
	def __repr__(self):
		if self[0] == -1:
			return self[1]
		else:
			return "%s_%s"%(self[0], self[1])

class rule():
	def __init__(self, lhs, A, B, op, PA={}, PB={}):
		self._lhs = lhs
		self._op = op
		self._A = set(A)
		self._B = set(B)
		self._PA = PA
		self._PB = PB

	def __repr__(self):
		#import pdb; pdb.set_trace()
		A = " | ".join(map(lambda x: "%s [%.3f]"%(str(x[0]), x[1]), self._PA.items()))
		B = " | ".join(map(lambda x: "%s [%.3f]"%(str(x[0]), x[1]), self._PB.items()))
		return "%s %s ( %s ) ( %s )"%(self._lhs, self._op, A, B)

	@staticmethod
	def fromBC(bc):
		col = np.sum(bc._table, axis=0)
		row = np.sum(bc._table, axis=1)
		PA, PB = {}, {}

		for i, r in enumerate(bc._rows):
			PA[r] = float(row[i]/bc._sum)
		for i, c in enumerate(bc._cols):
			PB[c] = float(col[i]/bc._sum)

		return rule(bc._nt, bc._rows, bc._cols, bc._op, PA, PB)

class Graph():
	# this is a data structure for multi-agent events

	def __init__(self, sample ):
		self._G = nx.DiGraph()
		m, n = sample.shape
		for i in range(m) :
			for j in range(n) :
				if sample[i][j]:
					nodex = (i, j)
					self._G.add_node(nodex, data=T((i, sample[i][j])), pos=nodex)
					if j+1 < n:
						nodey = (i, j+1)
						if sample[i][j+1]:
							self._G.add_node(nodey, data=T((i, sample[i][j+1])), pos=nodey)
							self.add_edge_noloop(nodex, nodey, type='<')
					for k in range(m):
						nodey = (k, j)
						if sample[k][j]:
							self._G.add_node(nodey, data=T((k, sample[k][j])), pos=nodey)
							self.add_edge_noloop(nodex, nodey, type='=')
	
	def add_edge_noloop(self, x, y, **attr):
		# wrapper to prevent signle loop 
		if x != y:
			self._G.add_edge(x, y, **attr)

	def reduction(self, R):
		# nt [op] A B
		A, B, nt, op = R._A, R._B, R._lhs, R._op
		G = self._G
		if op == '=':
			for (u,v,d) in G.edges(data=True):

				if d['type'] == '=' and G.node.has_key(u) and G.node.has_key(v):
					if G.node[u]['data'] in A and G.node[v]['data'] in B:
						self.merge(G, u, v, nt, '=')
					elif G.node[v]['data'] in B and G.node[u]['data'] in A:
						self.merge(G, v, u, nt, '=')
					
		elif op == "<":
			for (u,v,d) in G.edges(data=True):
				if d['type'] == '<' and G.node.has_key(u) and G.node.has_key(v):
					if G.node[u]['data'] in A and G.node[v]['data'] in B:
						self.merge(G, u, v, nt, '<')

	@staticmethod
	def merge(G, A, B, new, op):
		# merge two nodes A and B together to form new 
		#print "merge %s %s %s to %s"%(G.node[A]['data'], op, G.node[B]['data'], new)
		
		if op == '=':
			for (u,v,d) in G.in_edges([B], data=True):
				if u != A and u != B:
					G.add_edge(u, A, **d)
			for (u,v,d) in G.out_edges([B], data=True):
				if v != A and v != B :
					G.add_edge(A, v, **d)

		elif op == '<':
			for (u,v,d) in G.in_edges([B], data=True):
				if u != A and u != B and d['type'] != '=':
					G.add_edge(u, A, **d)
			for (u,v,d) in G.out_edges([B], data=True):
				if v != A and v != B and d['type'] != '=':
					G.add_edge(A, v, **d)

		G.add_node(A, data=new)
		G.remove_nodes_from([B])

	@staticmethod
	def _label(x):
		if x[0] == -1:
			return x[1]
		else:
			return "%s_%s"%(x[0], x[1])

	@staticmethod
	def _inverse(t):
		if t == "<":
			return '>'
		elif t == ">":
			return '<'
		else:
			return "="

	def _showT(self, t, r, c):
	    s =  "<\t"
	    s += "\t".join(map(str, c))
	    s += "\n"
	    for x in r:
	        s += "%s\t"%str(x)
	        for y in c:
	            s += "%s\t"%t[(x,y, '<')]
	        s += "\n"
	    s +=  "\n=\t"
	    s += "\t".join(map(str, c))
	    s += "\n"
	    for x in r:
	        s += "%s\t"%str(x)
	        for y in c:
	            s += "%s\t"%t[(x,y, '=')]
	        s += "\n"
	    return s

	def s2s(self):
		table = Counter()
		symbols = set()
		for (u,v,d) in self._G.edges_iter(data=True):
			x = self._G.node[u]['data']
			y = self._G.node[v]['data']
			op = d['type']
			table[(x, y, op)] += 1
			symbols.add(x)
			symbols.add(y)
		return table, symbols

	def ecm(self):
		table = Counter()
		rows = set()
		cols = set()
		for (n1, d) in self._G.nodes_iter(data=True):
			ns = self._G.successors(n1)
			ns.extend( self._G.predecessors(n1) )
			ns = list( set(ns) )
			nd1 = self._G.node[n1]['data']
			for (n2,n3) in itertools.permutations(ns, 2):
				d13 = self._G.get_edge_data(n1, n3)
				d31 = self._G.get_edge_data(n3, n1)
				nd2 = self._G.node[n2]['data']
				nd3 = self._G.node[n3]['data']
				if d13 :
					table[(nd1, nd2), nd3, d13['type']] += 1
					cols.add((n3, d13['type']))
				else :
					table[(nd1, nd2), nd3, self._inverse(d31['type'])] += 1
					cols.add((n3, self._inverse(d31['type'])))
				rows.add((n1, n2))
		return table, rows, cols
	



	def vis(self, file="test.png", rule=None):
		G = self._G
		edge1=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "="]
		edge2=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "<"]
		edge_labels = { (u,v) : d['type'] for (u,v,d) in G.edges(data=True)  }
		labels = {n:str(d['data']) for (n,d) in G.nodes(data=True)}
		initPos = { n:d['pos'] for (n,d) in G.nodes(data=True)}
		pos=nx.spring_layout(G, pos=initPos)
		nx.draw_networkx_nodes(G,pos=initPos,node_size=300, alpha=0.8)
		nx.draw_networkx_edges(G,pos=initPos,edgelist=edge1,
	                    width=2, alpha=0.8, arrows=False)
		nx.draw_networkx_edges(G,pos=initPos,edgelist=edge2,
	                    width=2, alpha=0.8, edge_color='b',style='dashed', arrows=True)
		nx.draw_networkx_labels(G,pos=initPos,labels=labels, font_size=8,font_family='sans-serif')
		#nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels, font_size=10,font_family='sans-serif')
		plt.axis('on')
		
		if rule:
			plt.title(str(rule))
		plt.xlabel("agents")
		plt.ylabel("time")
		plt.savefig(file,dpi=300)
		#plt.show()
		plt.clf()

def multi():
	sample = np.array(
		[['A', 'C', 'E'], 
		['A', 'D', 'D'],
		['B', 'C', 'D']]
		)
	G = Graph(sample)
	table, symbols = G.s2s()
	ecm, rows, cols = G.ecm()
	G.vis()

	r = rule("NT1", [T((0,'A'))], [T((1, 'A'))], '=')
	G.reduction(r)
	G.vis()

	r = rule("NT2", [T((0,'C'))], [T((2, 'C'))], '=')
	G.reduction(r)
	G.vis()

	r = rule("NT3", [T((1,'D'))], [T((2, 'D'))], '=')
	G.reduction(r)
	G.vis()
	
	r = rule("NT4", [T((1,'D'))], [T((-1, 'NT3'))], '<')
	G.reduction(r)
	G.vis()

	r = rule("NT5", [T((-1,'NT1'))], [T((2, 'B'))], '=')
	G.reduction(r)
	G.vis()

	r = rule("NT6", [T((-1,'NT2'))], [T((-1, 'NT4'))], '=')
	G.reduction(r)
	G.vis()

	r = rule("NT7", [T((-1,'NT5'))], [T((-1, 'NT6'))], '<')
	G.reduction(r)
	G.vis()

	r = rule("S", [T((-1,'NT7'))], [T((0, 'E'))], '<')
	G.reduction(r)

	G.vis()

def single():
	sample = np.array(
		[['A', 'B', 'A', 'C']]
		)
	G = Graph(sample)
	G.vis()

def learning(samples, alpha=0.05, beta=5, cut=30):
	from BiCluster import DupbestBC, BiCluster
	Gs = map( Graph, samples)
	bcs = []
	grammar = {}
	totalBits = sum(map(lambda x : x._G.order(), Gs))
	Gs[0].vis()
	for i in range(50):
		print "Compression:%s\n"%(sum(map(lambda x : x._G.order(), Gs))/float(totalBits))
		tables, symbols, ecms, cols = prepare(Gs)
		bc = DupbestBC(tables, symbols, ecms, cols, alpha=0.05, beta=5, cut=30)
		if not bc: 
			print "no more rules!"
			break
		#import pdb; pdb.set_trace()
		#print bc

		bcs.append(bc)
		new = T((-1, 'NT%s'%i))
		bc._nt = new
		r = rule.fromBC(bc)
		grammar[r._lhs] = r
		for G in Gs:
			G.reduction(r)

		tables, symbols, ecms, cols = prepare(Gs)
		for ind, _bc in enumerate(bcs):
			bc_new_c = BiCluster().update(_bc, tables, ecms, col=bc._nt)
			bc_new_r = BiCluster().update(_bc, tables, ecms, row=bc._nt)
			#print "bcG: %s"%bc_new.logGain()
			best = None
			if bc_new_c :
				bc_new = bc_new_c 
				best = bc_new_c.logGain()
			if bc_new_r and bc_new_r.logGain() > best:
				bc_new = bc_new_r
				best = bc_new_r.logGain()
			if best - bc.logGain() > 2.0:
				print _bc
				print "Attach"
				print bc_new
				r = rule.fromBC(bc_new)
				print r
				grammar[r._lhs] = r
				for G in Gs:
					G.reduction(r)
			bcs[ind] = bc_new
		#Gs[0].vis(file='big_%s.png'%str(new), rule = r)
	return Gs, grammar
def test1():
	samples = np.random.choice(['A','T','C','G', None], (3, 4,20))
	Gs, grammar = learning(samples, alpha=0.05, beta=5, cut=30)
	for prod in grammar.values():
		print prod

def big():
	from BiCluster import DupbestBC, BiCluster
	samples = np.random.choice(['A','T','C','G', None], (1, 4, 6))
	Gs = map( Graph, samples)
	bcs = []
	grammar = {}
	totalBits = sum(map(lambda x : x._G.order(), Gs))
	Gs[0].vis()
	for i in range(50):
		print "alpha:%s\n"%(sum(map(lambda x : x._G.order(), Gs))/float(totalBits))
		tables, symbols, ecms, cols = prepare(Gs)
		bc = DupbestBC(tables, symbols, ecms, cols)
		if not bc: 
			print "no more rules!"
			break
		#import pdb; pdb.set_trace()
		print bc

		bcs.append(bc)
		new = T((-1, 'NT%s'%i))
		bc._nt = new
		r = rule.fromBC(bc)
		print r
		grammar[r._lhs] = r
		for G in Gs:
			G.reduction(r)

		tables, symbols, ecms, cols = prepare(Gs)
		for ind, _bc in enumerate(bcs):
			bc_new_c = BiCluster().update(_bc, tables, ecms, col=bc._nt)
			bc_new_r = BiCluster().update(_bc, tables, ecms, row=bc._nt)
			#print "bcG: %s"%bc_new.logGain()
			best = None
			if bc_new_c :
				bc_new = bc_new_c 
				best = bc_new_c.logGain()
			if bc_new_r and bc_new_r.logGain() > best:
				bc_new = bc_new_r
				best = bc_new_r.logGain()
			if best - bc.logGain() > 2.0:
				print "Attach"
				print bc_new
				r = rule.fromBC(bc_new)
				print r
				grammar[r._lhs] = r
				for G in Gs:
					G.reduction(r)
			bcs[ind] = bc_new
		Gs[0].vis(file='big_%s.png'%str(new), rule = r)
	for g in grammar.values():
		print g

def prepare(Gs):
	tables = Counter()
	symbols = set()
	ecms = Counter()
	cols = set()
	for G in Gs:
		table, symbol = G.s2s()
		tables = tables + table
		symbols = symbols | symbol
		ecm, _, col = G.ecm()
		ecms = ecms + ecm
		cols = cols | col
	return tables, symbols, ecms, cols

def train():
	from BiCluster import DupbestBC, BiCluster
	samples = np.random.choice(['A','T','C','G', None], (3, 4,20))
	Gs = map( Graph, samples)
	bcs = []
	totalBits = sum(map(lambda x : x._G.order(), Gs))
	for i in range(10):
		print "alpha:%s\n"%(sum(map(lambda x : x._G.order(), Gs))/float(totalBits))
		tables, symbols, ecms, cols = prepare(Gs)
		bc = DupbestBC(tables, symbols, ecms, cols)
		if not bc: 
			print "no more rules!"
			break
		#import pdb; pdb.set_trace()
		print bc
		bcs.append(bc)
		new = T((-1, 'NT%s'%i))
		bc._nt = new
		r = rule(bc._nt, bc._rows, bc._cols, bc._op)
		print r
		for G in Gs:
			G.reduction(r)

		tables, symbols, ecms, cols = prepare(Gs)
		for _bc in bcs:
			
			bc_new = BiCluster().update(_bc, tables, ecms, col=new)
			#print "bcG: %s"%bc_new.logGain()
			if bc_new and bc_new.logGain() > 0.0:
				print "Adding col %s to %s"%(new, bc_new._nt)
				print bc_new
				#import pdb; pdb.set_trace()
				r = rule(bc_new._nt, bc_new._rows, bc_new._cols, bc_new._op)
				print r
				for G in Gs:
					G.reduction(r)
				
				bcs.append(bc_new)

			bc_new = BiCluster().update(_bc, tables, ecms, row=new)
			#print "bcG: %s"%bc_new.logGain()
			if bc_new and bc_new.logGain() > 0.0:
				print "Adding col %s to %s"%(new, bc_new._nt)
				print bc_new
				#import pdb; pdb.set_trace()
				r = rule(bc_new._nt, bc_new._rows, bc_new._cols, bc_new._op)
				print r
				for G in Gs:
					G.reduction(r)
				bcs.append(bc_new)

if __name__ == '__main__':
	test1()