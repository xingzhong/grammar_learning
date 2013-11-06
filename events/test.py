from event import *

def semanticMatrix(g):
	S = []
	for (x,y,d) in g.edges(data=True):
		S.append(np.hstack((x._semantics, y._semantics)))
	return np.array(S)

def vis2D(m, ax, means=None):
	
	ax.scatter(m[:,0], m[:,1], marker='o',c='b',s=5,zorder=10)
	#ax.plot(mean[0], mean[1], 'o')
	#ax.contourf(x1,y1,z,10,cmap=pl.cm.rainbow,
	#                  vmax=abs(z).max(), vmin=0)
	if not means is None:
		ax.scatter(means[:, 0], means[:, 1], marker='x', c='red', s=30)

	plt.grid()
	plt.show()

def gmm(samples, k=1):
	from sklearn.mixture import GMM
	cluster = GMM(n_components=k)
	cluster.fit(samples)
	return cluster

def make_ellipses(gmm, ax, idx):
	import matplotlib as mpl
	from matplotlib import cm
	k = gmm.get_params()['n_components']
	
	for n in range(k):
		if (idx is None) or n==idx:
			color = cm.jet(1.*n/k)
			v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
			u = w[0] / np.linalg.norm(w[0])
			angle = np.arctan2(u[1], u[0])
			angle = 180 * angle / np.pi  # convert to degrees
			v *= 9
			ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
				180 + angle, color=color)
			ell.set_clip_box(ax.bbox)
			ell.set_alpha(0.5)
			ax.add_artist(ell)

def vis(sm, cluster, k=None):
	fig = plt.figure(figsize=(8, 3))
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
	ax = fig.add_subplot(1, 1, 1)
	make_ellipses(cluster, ax, k)
	vis2D(sm, ax, means=cluster.means_)

def bestProd(sm, cluster, gamma=-4):
	logProds = log_multivariate_normal_density(sm, cluster.means_, cluster.covars_)
	idx = np.argmax( np.sum(logProds > gamma, axis=0) )
	coverage = np.sum(logProds > gamma, axis=0)
	print coverage
	print idx
	return idx, logProds[:, idx]

def toProd(graph, gamma=-4, k=10):
	sm = semanticMatrix(g)
	cluster = gmm(sm, k=k)
	idx, logProd = bestProd(sm, cluster, gamma=gamma)
	nx.set_edge_attributes(graph, "delta", dict(zip(graph.edges(), logProd)))
	return cluster.means_[idx], cluster.covars_[idx]

def rewrite(graph, means, covars, gamma=-4):
	edges = []
	B = set()
	C = set()
	for (x,y,d) in graph.edges(data=True):
		if d['delta'] > gamma:
			edges.append( (x, y) )
			B.add(x)
			C.add(y)
	nx.set_node_attributes(graph, "cluster", dict(zip(B, ["blue"]*len(B))))
	nx.set_node_attributes(graph, "cluster", dict(zip(C, ["red"]*len(C))))
	print means
	print covars
	drawG2(graph, cluster=True, label=True)


if __name__ == '__main__':
	g = EventGraph()
	left = norm(loc=np.array([4.0]))
	right = norm(loc=np.array([-4.0]))
	stop = norm(loc=np.array([0.0]))
	#sample = np.random.choice([left, right, stop, None], size=(4,6), p=[0.3,0.3,0.3,0.1])
	sample = np.random.choice([left, right, stop, None], size=(6,10), p=[0.4,0.25,0.25,0.1])
	rvs = np.frompyfunc(lambda x:x.rvs() if x else None, 1, 1)
	samples = rvs(sample)
	for aid, seq in enumerate (sample):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom.rvs()))
	g.buildEdges(delta = 1)
	means, covars = toProd(g)
	rewrite(g, means, covars)
