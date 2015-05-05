import pymc
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib as mpl

ndata = 30
n = 3

def sampleData():
	return np.array([[20, 20], [20, 40], [20, 60], [20, 80],
						[50, 25], [50, 45], [55, 52], [45, 75],
						[70, 50], [80, 70]])

def model(X):
	dd = pymc.Dirichlet('dd', theta=(1,)*n)
	category = pymc.Categorical('category', p=dd, size=ndata)

	#precs = pymc.Gamma('precs', alpha=2, beta=1, size=n)
	#means = pymc.Uniform('means', lower=-30, upper=30, size=n)
	xm = pymc.Uniform('xm', lower=-20, upper=20, size=n)
	ym = pymc.Uniform('ym', lower=-20, upper=20, size=n)

	means = np.empty(n, dtype=object)
	precs = np.empty(n, dtype=object)
	for i in range(n):
		means[i] = pymc.MvNormal('means_%i'%i, np.ones(2), np.eye(2))
		precs[i] = pymc.Gamma('precs_%i'%i, alpha=2, beta=1) * np.eye(2)

	@pymc.deterministic
	def mean(category=category, means=means, xm=xm, ym=ym):
		#return np.ones(2)
		#return xm[category] * np.ones(2)
		return means[category] * np.ones(2)

	@pymc.deterministic
	def prec(category=category, precs=precs):
		return precs[category]

	#E_dp = pm.Lambda('E_dp', lambda p=dd, theta=theta: np.dot(p, theta))

	obs = pymc.MvNormal('obs', mean, np.eye(2), value=X, observed = True)
	return locals()

def chinese_restaurant_process(n, alpha):
    
    if n < 1:
        return None
    
    table_assignments = np.empty(n)
    next_table = 0
    
    for c in range(n):

        if np.random.random() < (1. * alpha / (alpha + c)):
            
            # Sit at new table
            table_assignments[c] = next_table
            next_table += 1
            
        else:
            
            # Calculate selection probabilities as function of population
            probs = [(table_assignments[:c]==i).sum()/float(c) 
                     for i in range(next_table)]
            # Randomly assign to existing table
            table_assignments[c] = choice(range(next_table), p=probs)
        
    return table_assignments


def vModel(X):
	nDim, dDim = X.shape

	dd = pymc.Dirichlet('dd', theta=(1,)*n)
	
	#centers = pymc.Container([
	#			pymc.Uniform('means%i'%i, lower=-30, upper=30, size=dDim) 
	#			for i in range(nDim)])
	centers = pymc.Uniform('means', lower=0, upper=100, size=(n, dDim)) 
	center = pymc.Normal('center', [50, 50], np.diag([1, 0.5]))
	#sigmasX = pymc.Gamma('sigmas', alpha=1, beta=1, size=n)
	sigmasX = pymc.Normal('sigmasX', mu=20, tau=1, size=n)
	sigmasY = pymc.Normal('sigmasY', mu=60, tau=1, size=n)

	category = pymc.Container([pymc.Categorical("category%i" % i, p=dd) 
				for i in range(nDim)])
	#@pymc.deterministic
	#def mean(category=category, xm=xm, ym=ym):
#		return np.array([xm[category], ym[category]])

	observations = pymc.Container([pymc.MvNormal('samples_model%i' % i, 
                   mu=centers[category[i],:], 
                   tau= np.diag([sigmasX[category[i]], sigmasY[category[i]]]), 
                   value=X[i], observed=True) for i in range(nDim)])
	return locals()

def sModel(X):
	nDim, dDim = X.shape

	center1 = pymc.MvNormal('center1', [50, 50], np.diag([1, 0.5]))
	sigma1  = pymc.MvNormal('sigma1',  [20, 60], np.diag([1 ,1.0 ]))
	center2 = pymc.MvNormal('center2', [25, 50], np.diag([1, 0.5]))
	sigma2  = pymc.MvNormal('sigma2',  [20, 60], np.diag([1 ,1.0 ]))

	w = pymc.Uniform('weight', 0, 1)
	#sigmasX = pymc.Normal('sigmasX', 20, 10)
	#sigmasY = pymc.Normal('sigmasY', 60, 10)

	obs1 = pymc.MvNormal('obs1', 
                   mu= center1, 
                   tau= np.diag(sigma1.value), 
                   value=X, observed=True).logp

	obs2 = pymc.MvNormal('obs2', 
                   mu= center2, 
                   tau= np.diag(sigma2.value), 
                   value=X, observed=True).logp

	obs = w*obs1 + (1-w) * obs2
	return locals()

if __name__ == '__main__':
	x = sampleData()
	#iris = datasets.load_iris()
	#x = iris.data[:, :2]
	#plt.scatter(x[:,0], x[:,1])
	colors = ['#800080', '#66CDAA', '#DC143C', '#808000', '#00FFFF']
	#print chinese_restaurant_process(10, 1)
	m = pymc.MCMC(sModel( x ))
	#m.centers.value = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]])
	#m.use_step_method(pymc.AdaptiveMetropolis, m.means)
	#m.use_step_method(pymc.AdaptiveMetropolis, m.precs)
	#import ipdb; ipdb.set_trace()
	m.center1.value = np.array([50, 50])
	m.center2.value = np.array([25, 50])
	m.sample(iter=10000, burn=500, thin=3)
	#plt.hist(x)
	#import ipdb; ipdb.set_trace()
	#c = map(lambda x:colors[x], m.category.value)
	#pymc.Matplot.plot(m)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x[:,0], x[:,1], s=40, c='red')

	ell = mpl.patches.Ellipse(m.center1, m.sigma1[0], m.sigma1[1], 0, color=colors[0])
	ell.set_clip_box(ax.bbox)
	ell.set_alpha(0.5)
	ax.add_artist(ell)

	ell = mpl.patches.Ellipse(m.center2, m.sigma2[0], m.sigma2[1], 0, color=colors[1])
	ell.set_clip_box(ax.bbox)
	ell.set_alpha(0.5)
	ax.add_artist(ell)
	plt.grid()
	plt.show()

	print m.centers.value
	print m.sigmasX.value
	print m.sigmasY.value

	for n, color in enumerate(colors[:n]):
		#import ipdb; ipdb.set_trace()
		w = m.sigmasX.value[n]
		v = m.sigmasY.value[n]
		ell = mpl.patches.Ellipse(m.centers.value[n, :2], w, v, 0, color=color)
		ell.set_clip_box(ax.bbox)
		ell.set_alpha(0.5)
		ax.add_artist(ell)

	#plt.scatter(m.centers.value[:,0], m.centers.value[:,1], s=81*m.sigmas.value**2, c='r', alpha=0.3)
	plt.grid()
	plt.show()
	#pymc.Matplot.plot(m.centers)
	#plt.show()
	#