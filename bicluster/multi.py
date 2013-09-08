import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools

def replace(G, old, new):
	print "%s -> %s"%(old, new)
	G.add_node(new, data=(-1, new))
	for p,v in G[old].items():
		G.add_edge(p, new, **v)
	for p in G.neighbors(old):
		G.remove_edge(p, old)
	G.remove_node(old)

def reduction(G, rows, cols, nt, op):
	# given a graph data G,
	# rows and cols are merge set
	# nt is new non-terminal
	# N1 = 1A 2A
	if op == '=':
		for (u,v,d) in G.edges(data=True):
			if d['type'] == '=' and G.node.has_key(u) and G.node.has_key(v):
				if G.node[u]['data'] in rows and G.node[v]['data'] in cols:
					replace(G, u, nt )
					replace(G, v, nt )
				elif G.node[v]['data'] in rows and G.node[u]['data'] in cols:
					replace(G, u, nt )
					replace(G, v, nt )
				
	elif op == "<":
		for (u,v,d) in G.edges(data=True):
			if d['type'] == '<' :
				if G.node[u]['data'] in rows and G.node[v]['data'] in cols:
					replace(G, u, nt )
					replace(G, v, nt )

def main():
	sample = np.array(
		[['A', 'C', 'E'], 
		['A', 'D', 'D'],
		['B', 'C', 'D']]
		)
	G = nx.Graph()
	m, n = sample.shape
	for i in range(m) :
		for j in range(n) :
			nodex = (i, j)
			G.add_node(nodex, data=(i, sample[i][j]))
			if j+1 < n:
				nodey = (i, j+1)
				G.add_node(nodey, data=(i, sample[i][j+1]))
				G.add_edge(nodex, nodey, type='<')
			for k in range(m):
				nodey = (k, j)
				G.add_node(nodey, data=(k, sample[k][j]))
				G.add_edge(nodex, nodey, type='=')
	#draw(G, sample)
	cols = set()
	cols.add((0, 'A'))
	rows = set()
	rows.add((1, 'A'))
	reduction(G, cols, rows, 'NT1', '=')
	draw(G, sample)

	cols = set()
	cols.add((0, 'C'))
	rows = set()
	rows.add((2, 'C'))
	reduction(G, cols, rows, 'NT2', '=')
	draw(G, sample)

	cols = set()
	cols.add((1, 'D'))
	rows = set()
	rows.add((2, 'D'))
	reduction(G, cols, rows, 'NT3', '=')
	draw(G, sample)

	cols = set()
	cols.add((-1, 'NT3'))
	rows = set()
	rows.add((1, 'D'))
	reduction(G, cols, rows, 'NT4', '<')
	draw(G, sample)

	cols = set()
	cols.add((-1, 'NT1'))
	rows = set()
	rows.add((2, 'B'))
	reduction(G, cols, rows, 'NT5', '=')
	draw(G, sample)

	cols = set()
	cols.add((-1, 'NT2'))
	rows = set()
	rows.add((-1, 'NT4'))
	reduction(G, cols, rows, 'NT6', '=')
	draw(G, sample)

	cols = set()
	cols.add((-1, 'NT6'))
	rows = set()
	rows.add((-1, 'NT5'))
	reduction(G, cols, rows, 'NT7', '<')
	draw(G, sample)
	
	cols = set()
	cols.add((0, 'E'))
	rows = set()
	rows.add((-1, 'NT7'))
	reduction(G, cols, rows, 'S', '<')
	draw(G, sample)

def draw(G, S):
	edge1=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "="]
	edge2=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "<"]
	edge_labels = { (u,v) : d['type'] for (u,v,d) in G.edges(data=True)  }
	labels = {n:str(d['data']) for (n,d) in G.nodes(data=True)}
	pos=nx.spring_layout(G)
	nx.draw_networkx_nodes(G,pos,node_size=2000)
	nx.draw_networkx_edges(G,pos,edgelist=edge1,
                    width=6)
	nx.draw_networkx_edges(G,pos,edgelist=edge2,
                    width=6,alpha=0.5,edge_color='b',style='dashed')
	nx.draw_networkx_labels(G,pos,labels=labels, font_size=10,font_family='sans-serif')
	nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels, font_size=10,font_family='sans-serif')
	plt.axis('off')
	plt.show()
	#plt.savefig("test.png",dpi=75)
if __name__ == '__main__':
	main()