from event import *
try :
	np.random.choice(5)
	from numpy.random import choice as choice
except :
	print "using own choice"
	def choice(x, size=1, p=None) :
		idx = np.random.randint(0, high=len(x), size=size)
		choose = np.frompyfunc( lambda y: x[y], 1, 1 )
		return choose(idx)

def semanticMatrix(g):
    left = []
    right = []
    for (x,y,d) in g.edges(data=True):
        left.append(x._semantics)
        right.append(y._semantics)
    n = max( map( lambda x: x.shape[0], left ) )
    for i,l in enumerate(left):
        left[i] = np.append(l, [0]*(n-l.shape[0]))
    n = max( map( lambda x: x.shape[0], right ) )
    for i,r in enumerate(right):
        right[i] = np.append(r, [0]*(n-r.shape[0]))
    left = np.array(left)
    right = np.array(right)
    return np.hstack((left, right))

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

def make_ellipses(model, ax, idx):
	import matplotlib as mpl
	from matplotlib import cm
	k = model.get_params()['n_components']
	for n in range(k):
		if (idx is None) or n==idx:
			color = cm.jet(1.*n/k)
			v, w = np.linalg.eigh(model._get_covars()[n][:2, :2])
			u = w[0] / np.linalg.norm(w[0])
			angle = np.arctan2(u[1], u[0])
			angle = 180 * angle / np.pi  # convert to degrees
			v *= 9
			ell = mpl.patches.Ellipse(model.means_[n, :2], v[0], v[1],
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

def bestProd(sm, cluster):
	idx = np.argmax(cluster.weights_)
	logProds = log_multivariate_normal_density(sm, np.array([cluster.means_[idx]]), np.array([cluster.covars_[idx]]))
	return idx, logProds

def toProd(graph, gamma=-4, k=10):
	sm = semanticMatrix(g)
	n = sm.shape
	print n, k
	cluster = gmm(sm, k=min(k, n[0]))
	idx, logProd = bestProd(sm, cluster)
	nx.set_edge_attributes(graph, "delta", dict(zip(graph.edges(), logProd)))
	print cluster.means_[idx], cluster.covars_[idx]
	#vis(sm, cluster, k=None)
	return cluster.means_[idx], cluster.covars_[idx]

def rewrite(graph, means, covars, gamma=-4):
	edges = filter(lambda x: x[2]['delta'] > gamma, graph.edges(data=True))
	if len(edges) > 0:
		for (x,y,d) in sorted(edges, key=lambda x:x[2]['delta'], reverse=True) :
			if graph.has_node(x) and graph.has_node(y) :
				semantics = np.mean( means.reshape((2, len(means)/2)), axis=0 )
				nt = Event(-1, x._aids | y._aids, means)
				graph._merge(x, y, nt, d)
				rewrite(graph, means, covars, gamma=gamma)

	


if __name__ == '__main__':
	g = EventGraph()
	left = norm(loc=np.array([5.0]), scale=0.5)
	right = norm(loc=np.array([-5.0]), scale=0.5)
	stop = norm(loc=np.array([0.0]), scale=0.5)
	#sample = np.random.choice([left, right, stop, None], size=(4,6), p=[0.3,0.3,0.3,0.1])
	sample = choice([left, right, stop, None], size=(5,9), p=[0.4,0.25,0.25,0.1])
	rvs = np.frompyfunc(lambda x:x.rvs() if x else None, 1, 1)
	samples = rvs(sample)

	for aid, seq in enumerate (sample):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom.rvs() ))

	g.buildEdges(delta = 1)

	for i in range (5) :
		if len(g.nodes()) < 3 :
			break
		means, covars = toProd(g)
		#drawG2(g, node_size=1400, cluster=False, label=True, output="test_%s"%i, 
		#		title="%s"%(means.reshape(2, len(means)/2)))
		
		rewrite(g, means, covars, gamma=-5)
