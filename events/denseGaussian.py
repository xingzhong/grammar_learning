import numpy as np
import pylab as pl

from sklearn.mixture import log_multivariate_normal_density
from scipy import optimize

def normal(x, mu):
	means = np.array([mu])
	m,n = x.shape
	covars = 0.01*np.ones((1,n))
	return log_multivariate_normal_density(x, means, covars)

def adjust(x, mu):
	means = np.array([mu])
	m,n = x.shape
	covars = 0.01*np.ones((1,n))
	p = log_multivariate_normal_density(x, means, covars)
	return np.sum((p > -0.5 )* 1)

def select(x, mu):
	means = np.array([mu])
	m,n = x.shape
	covars = 0.01*np.ones((1,n))
	p = log_multivariate_normal_density(x, means, covars)
	return p > -0.5

#print minimize(lambda x: -adjust(samples, x), mean, method='BFGS' )
#print res.x
#print res.message
def estimate2(samples):
	N = 50
	x1 = np.linspace(min(samples[:,0]), max(samples[:,0]), N)
	y1 = np.linspace(min(samples[:,1]), max(samples[:,1]), N)
	z = np.zeros((N,N))
	for j, x0 in enumerate(y1):
		for i, y0 in enumerate(x1):
			z[i,j] = adjust(samples, [x0, y0])
			if z[i,j]>10:
				print x0, y0, z[i,j]
	maxx, maxy = np.unravel_index( np.argmax(z), (N,N) )
	print x1[maxx], y1[maxy], z[maxx, maxy]
	return x1, y1, z, x1[maxx], y1[maxy]
#print normal(samples, [x1[maxx], y1[maxy]])
def estimate(samples):
	N = 10
	MAXS = np.max(samples, axis = 0)
	MINS = np.min(samples, axis = 0)
	#print MINS, adjust(samples, MINS)
	#print MAXS, adjust(samples, MAXS)
	rrange = zip(MINS, MAXS)
	res = optimize.brute(lambda x: -adjust(samples, x), 
			rrange,  Ns=N, full_output=False)
	print res, adjust(samples, res)
	return res, select(samples, res)

if __name__ == '__main__':
	samples = np.random.multivariate_normal([-1,1], 0.1*np.eye(2), 50)
	samples = np.vstack( (samples, np.random.multivariate_normal([0,0], 0.4*np.eye(2), 50)))
	samples = np.vstack( (samples, np.random.multivariate_normal([-1,-1], 0.1*np.eye(2), 50) ))
	samples = np.vstack( (samples, np.random.multivariate_normal([1,1], 0.1*np.eye(2), 50) ) )

	x1, y1, z, x0, y0 = estimate2(samples)
	print select(samples, [x0,y0])
	#estimate(samples)

	fig = pl.figure(figsize=(8, 3))
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
	ax = fig.add_subplot(1, 1, 1)

	ax.scatter(samples[:,0], samples[:,1], marker='o',c='b',s=5,zorder=10)
	#ax.plot(mean[0], mean[1], 'o')
	ax.contourf(x1,y1,z,10,cmap=pl.cm.rainbow,
	                  vmax=abs(z).max(), vmin=0)

	pl.show()