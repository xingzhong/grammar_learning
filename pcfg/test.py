from sklearn.mixture import log_multivariate_normal_density
import numpy as np 

def _log_multivariate_normal_density_diag(X, means=0.0, covars=1.0):
	"""Compute Gaussian log-density at X for a diagonal model"""
	n_samples, n_dim = X.shape
	#import pdb; pdb.set_trace()
	lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
	return lpr

def dmvnorm(b, mean, cov):
	k = b.shape[0]
	part1 = np.exp(-0.5*k*np.log(2*np.pi))
	part2 = np.power(np.linalg.det(cov),-0.5)
	dev = b-mean
	part3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(cov)),dev))
	dmvnorm = part1*part2*part3
	return dmvnorm



X = np.array([0.2,1,0.1])
mean = np.array([0,1,1])
covar = np.identity(3)

print X, mean, covar
print dmvnorm(X, mean, covar)
