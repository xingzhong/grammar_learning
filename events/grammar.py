from sklearn.mixture import log_multivariate_normal_density, sample_gaussian
class NT():
	def __init__(self, mean, covar, name):
		idx = np.max( np.where( np.abs(mean)>1e-6))
		self.mean = mean[:idx+1]
		self.covar = covar[:idx+1]
		self.name = name

	def setName(self, name):
		self.name = name

	def __repr__(self):
		return str(self.name) + str(self.mean)

	def logPdf(self, x):
		x = np.array([x], ndmin=2)
		return log_multivariate_normal_density(x, self.mean, self.covar)

	def sample(self):
		return sample_gaussian(self.mean, self.covar)

class Prod(object) :
	def __init__(self, means, covars, type="<", idx=0):
		self._type = type
		self._means = means.reshape(2, len(means)/2)
		self._covars = covars.reshape(2, len(covars)/2)
		self._idx = idx
		self._name = "NT%s"%self._idx
		self.build()

	def build(self):
		# A -> B C
		self.B = NT(self._means[0, :], self._covars[0, :], name='NT%s_1'%self._idx)
		self.C = NT(self._means[1, :], self._covars[1, :], name='NT%s_2'%self._idx)
	
	def toRewrite(self):
		return np.concatenate( (np.ravel(self.B.mean) , np.ravel(self.C.mean) ))

	def __repr__(self):
		return "%s -> %s %s"%(self._name, self.B, self.C)

class PCFG(object):
	def __init__(self):
		self.prods = []


	def nts(self):
		return [ (p.B, p.C, p._name) for p in self.prods]
		#return sum(nts, []) #flatten the 2 dim list

	def add(self, x):
		for (b, c, name) in self.nts():
			nt = np.concatenate( (np.ravel(b.mean) , np.ravel(c.mean) ))
			ntB = np.ravel(x.B.mean)
			ntC = np.ravel(x.C.mean)
			if len(nt) == len(ntB) and np.allclose(nt, ntB):
				x.B.setName(name)
			if len(nt) == len(ntC) and np.allclose(nt, ntC):
				x.C.setName(name)
		self.prods.append(x)

if __name__ == '__main__':
	from hmm import *

	np.set_printoptions(precision=2)
	X, dates, close_v = sample2()
	g = graph(X)
	gamma = -5
	N = 8
	figsize=(18,8)
	fig = pl.figure(figsize=figsize)
	grammar = PCFG()

	for i in range (N) :
		ax = fig.add_subplot(N, 1, i+1)
		ax.set_title(i+1)
		drawG3(ax, g, X)
		means, covars = toProd(g, k=10, gamma=gamma)
		#means, covars = cutDim(means, covars)
		p = Prod(means, covars, idx=i)
		grammar.add(p)
		rewrite(g, p.toRewrite(), None, gamma=gamma)
	
	for nt in grammar.prods:
		print nt
	plt.show()
	
