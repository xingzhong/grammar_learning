import numpy as np
from sklearn.mixture import GMM
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

class Prod:
	def __init__(self, lhs, rhs1, rhs2, op='='):
		self.lhs = lhs
		self.left = rhs1
		self.ldim = self.left._nd
		self.right = rhs2
		self.rdim = self.right._nd
		self.op = op

	def __repr__(self):
		return "%s -> %s %s"%(self.lhs, self.left, self.right)



def sample():
	g = EventGraph()
	left = norm(loc=np.array([5.0]), scale=0.1)
	right = norm(loc=np.array([-5.0]), scale=0.1)
	stop = norm(loc=np.array([0.0]), scale=0.1)
	#sample = np.random.choice([left, right, stop, None], size=(4,6), p=[0.3,0.3,0.3,0.1])
	#sample = choice([left, right, stop, None], size=(1,15), p=[0.4,0.4,0.2,0.0])
	sample = [[left, left, left, stop, stop, right, right, right] * 2]
	for aid, seq in enumerate (sample):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom.rvs() ))
	g.buildEdges(delta = 1)
	return g

def semanticMatrix(g):
    left = []
    right = []
    for (x,y,d) in g.edges(data=True):
        left.append(x._semantics)
        right.append(y._semantics)
    if len(left) > 0:
    	m = max( map( lambda x: x.shape[0], left ) )
    else :
    	m = 0
    if len(right) > 0:
    	n = max( map( lambda x: x.shape[0], right ) )
    else :
    	n = 0

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

def gmm(samples, k=1):
	cluster = GMM(n_components=k)
	cluster.fit(samples)
	return cluster

def bestProd(sm, cluster):
	# select the maximum weight cluster
	idx = np.argmax(cluster.weights_)
	logProds = log_multivariate_normal_density(sm, np.array([cluster.means_[idx]]), np.array([cluster.covars_[idx]]))
	return idx, logProds

def trim(mean, covar=None):
	idx = np.flatnonzero(mean)[-1]
	if covar is not None:
		return mean[:idx+1], covar[:idx+1]
	else:
		return mean[:idx+1]

def toProd(graph, gamma=-4, k=10):
	# given data graph, extract a vaild production
	sm = semanticMatrix(graph)
	n = sm.shape
	print n
	print sm
	cluster = gmm(sm, k=min(k, n[0]))
	idx, logProd = bestProd(sm, cluster)
	means, covars = cluster.means_[idx], cluster.covars_[idx]
	lMean, rMean = means[:len(means)/2], means[len(means)/2:]
	lCov, rCov = covars[:len(covars)/2], covars[len(covars)/2:]
	lMean, lCov = trim(lMean, lCov)
	rMean, rCov = trim(rMean, rCov)
	
	#import pdb; pdb.set_trace()
	#print 'debug', lMean, rMean
	div = Dist(lMean, lCov) - Dist(rMean, rCov)
	
	if div > 1 :
		means = np.hstack((lMean, rMean))
		covs = np.hstack((lCov, rCov))
		#print 'two', means
		return Prod( Dist(means, covs), Dist(lMean, lCov), Dist(rMean, rCov) )
	else:
		#means = np.mean( np.vstack((lMean, rMean)), axis=0)
		#print 'single', means
		return Prod( Dist(lMean, lCov), Dist(lMean, lCov), Dist(rMean, rCov) )

def parse(graph, prod, gamma=-4):
	edges = []
	for (x,y,d) in graph.edges(data=True):
		#print x._semantics, x._semantics.shape, prod.left, prod.ldim
		#print y._semantics, y._semantics.shape, prod.right, prod.rdim
		if x._semantics.shape == prod.ldim and y._semantics.shape == prod.rdim:
			logLeft = prod.left.logPdf(x._semantics)
			logRight = prod.right.logPdf(y._semantics)
			logP = logLeft + logRight
			if logP > gamma:
				edges.append( (x,y,d))
	edges = sorted(edges, key=lambda x: x[0]._tp)
	return edges

def rewrite(graph, prod, gamma=-4):
	edges = parse(graph, prod, gamma=gamma)
	#pprint (edges)
	if len(edges) > 0:
		for (x,y,d) in edges :
			if graph.has_node(x) and graph.has_node(y) :
				nt = Event(-1, x._aids | y._aids, prod.lhs._mean )
				graph._merge(x, y, nt, d)
				#break
				rewrite(graph, prod, gamma=gamma)

def VisGraph(G, ax, node_size=1600, offset=0, edge=True, label=True):
    
    edge1=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "="]
    edge2=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "<"]
   
    #labels = {n:str(d) for (n,d) in G.nodes(data=True)}
    initPos = { n:(n._tp , np.mean( map(float, n._aids ) ) + offset  ) 
        for (n,d) in G.nodes(data=True)}
    pos=nx.spring_layout(G, pos=initPos)      
    labels = { n : formartSemantic(n._semantics) for n in G.nodes() }
    colors = 'white'
    #import pdb; pdb.set_trace()
    nx.draw_networkx_nodes(G,pos=initPos, node_size=node_size, alpha=0.8, node_color=colors, node_shape='o', cmap=plt.get_cmap('Accent'))
    if edge:
        nx.draw_networkx_edges(G,pos=initPos,edgelist=edge1,
                        width=1, alpha=0.6, arrows=False, ax=ax)
        nx.draw_networkx_edges(G,pos=initPos,edgelist=edge2,
                        width=1, alpha=0.6, edge_color='b', arrows=True, ax=ax)
    if label:
        nx.draw_networkx_labels(G, ax=ax, pos=initPos, labels = labels, font_size=14,font_family='sans-serif',label_pos=0.8)
        #edge_labels = { (u,v) : "%.2f"%d[weight] for (u,v,d) in G.edges(data=True) }
        #nx.draw_networkx_edge_labels(G, ax=ax, pos=initPos, edge_labels=edge_labels, font_size=10,font_family='sans-serif', alpha=0.5)
    
class Grammar ():
	def __init__(self):
		self._prods = []
		self._nts = {}
		self.num = 0

	def addNT(self, dist):
		for key, item in self._nts.iteritems():
			if item - dist < 1:
				return key
		self.num += 1
		self._nts["NT%s"%self.num] = dist
		return "NT%s"%self.num

	def addProd(self, prod):
		left = self.addNT(prod.left)
		right = self.addNT(prod.right)
		lhs = self.addNT(prod.lhs)
		self._prods.append( (lhs, left, right) )

def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod()
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum()          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum() # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N



class Dist():
	def __init__(self, mean, covar):
		self._mean = np.array(mean)
		self._covar = np.array(covar) 
		self._nd = mean.shape

	def sample(self):
		return sample_gaussian(self._mean, self._covar)

	def logPdf(self, x):
		x = np.array([x], ndmin=2)
		return log_multivariate_normal_density(x, np.array([self._mean]), np.array([self._covar]))

	def __sub__(self, other):
		if self._nd == other._nd :
			return gau_kl(self._mean, self._covar, other._mean, other._covar)
		else:
			return np.inf
	def __repr__(self):
		return "(%s %s)"%(str(self._mean), str(self._covar))


if __name__ == '__main__':
	figsize=(30,8)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(1, 1, 1)
	np.set_printoptions(precision=3)
	g = sample()
	gamma = -8
	grammar = Grammar()
	for i in range(3):
		print i
		VisGraph(g, ax, offset=i)
		if len(g.nodes()) < 3:
			break
		
		p = toProd(g, k=8, gamma=gamma)
		print p
		grammar.addProd(p)
		rewrite(g, p, gamma=gamma)
	
	pprint (grammar._nts)
	pprint (grammar._prods)
	plt.show()
	
