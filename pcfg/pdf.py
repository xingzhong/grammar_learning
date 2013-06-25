from scipy.stats import norm 

class Dist(object):
	def __init__(self, mean, var):
		self.mean = mean
		self.var = var 
		self.dist = norm(loc=mean, scale=var)

	def logpdf(self, x):
		#print self.mean, x, self.dist.logpdf(x)
		return self.dist.logpdf(x)

	def rvs(self, x):
		return self.dist.rvs(x)



if __name__ == '__main__':
	x = norm(loc=0, scale=1)
	print x.rvs(size=10)
	print x.pdf(0.0)