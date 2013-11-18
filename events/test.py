from event import *
from kinetic import kinetic
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
    m = max( map( lambda x: x.shape[0], left ) )
    n = max( map( lambda x: x.shape[0], right ) )
    N = max(m, n)
    for i,l in enumerate(left):
        left[i] = np.append(l, [0]*(N-l.shape[0]))
    for i,r in enumerate(right):
        right[i] = np.append(r, [0]*(N-r.shape[0]))
    left = np.array(left)
    right = np.array(right)
    #print left
    #print right
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
			
			#v, w = np.linalg.eigh(model._get_covars()[n][[[2],[5]], [2,5]])
			v, w = np.linalg.eigh(model._get_covars()[n][:2, :2])

			u = w[0] / np.linalg.norm(w[0])
			angle = np.arctan2(u[1], u[0])
			angle = 180 * angle / np.pi  # convert to degrees
			v *= 9
			ell = mpl.patches.Ellipse(model.means_[n, :2], v[0], v[1],
				180 + angle, color=color)
			#ell = mpl.patches.Ellipse(model.means_[n, [2,5]], v[0], v[1],
			#	180 + angle, color=color)
			ell.set_clip_box(ax.bbox)
			ell.set_alpha(0.5)
			ax.add_artist(ell)

def vis(sm, cluster, k=None):
	fig = plt.figure(figsize=(8, 3))
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
	ax = fig.add_subplot(1, 1, 1)
	make_ellipses(cluster, ax, k)
	vis2D(sm, ax)

def bestProd(sm, cluster):
	idx = np.argmax(cluster.weights_)
	logProds = log_multivariate_normal_density(sm, np.array([cluster.means_[idx]]), np.array([cluster.covars_[idx]]))
	return idx, logProds

def toProd(graph, gamma=-4, k=10):
	sm = semanticMatrix(graph)
	n = sm.shape
	print n, k
	#print sm
	cluster = gmm(sm, k=min(k, n[0]))
	idx, logProd = bestProd(sm, cluster)
	nx.set_edge_attributes(graph, "delta", dict(zip(graph.edges(), logProd)))
	edges = filter(lambda x: x[2]['delta'] > gamma, graph.edges(data=True))
	c = {}
	for edge in edges :
		c[edge[0]] = 'pink'
		c[edge[1]] = 'pink'
	nx.set_node_attributes(graph, 'cluster', c)
	
	#vis(sm, cluster, k=idx)
	return cluster.means_[idx], cluster.covars_[idx]

def rewrite(graph, means, covars, gamma=-4):
	edges = filter(lambda x: x[2]['delta'] > gamma, graph.edges(data=True))
	
	if len(edges) > 0:
		for (x,y,d) in sorted(edges, key=lambda x:x[2]['delta'], reverse=True) :
			if graph.has_node(x) and graph.has_node(y) :
				nt = Event(-1, x._aids | y._aids, means )
				graph._merge(x, y, nt, d)
				rewrite(graph, means, covars, gamma=gamma)

def cutDim(x, y) :
	#print 'old', x
	xx = x.reshape(2, len(x)/2)
	yy = y.reshape(2, len(x)/2)
	idx = np.max( np.where( np.any(np.abs(xx)>1e-10, axis=0) ))
	xxx = xx[:, :idx+1].flatten()
	yyy = yy[:, :idx+1].flatten()
	#import pdb; pdb.set_trace()
	#print 'new', xxx
	return xxx, yyy
	

def sample():
	g = EventGraph()
	left = norm(loc=np.array([5.0]), scale=0.1)
	right = norm(loc=np.array([-5.0]), scale=0.1)
	stop = norm(loc=np.array([0.0]), scale=0.1)
	#sample = np.random.choice([left, right, stop, None], size=(4,6), p=[0.3,0.3,0.3,0.1])
	sample = choice([left, right, stop, None], size=(1,15), p=[0.4,0.4,0.2,0.0])
	
	for aid, seq in enumerate (sample):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom.rvs() ))
	g.buildEdges(delta = 1)
	return g

if __name__ == '__main__':
	np.set_printoptions(precision=5)
	g = sample()
	#g = kinetic(M=0, N=10)
	#g = kinetic()
	for i in range (5) :
		if len(g.nodes()) < 2 :
			break
		means, covars = toProd(g, k=10, gamma=-5)
		print means
		print means.reshape(2, len(means)/2)

		drawG2(g, node_size=2000, cluster=True, label=True, output="test_%s"%i, 
				title="%s"%(means.reshape(2, len(means)/2)))
		means, covars = cutDim(means, covars)
		rewrite(g, means, covars, gamma=-5)
