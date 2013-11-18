from event import *
from test import toProd, rewrite, cutDim, semanticMatrix
import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.hmm import GaussianHMM
from numpy.random import choice as choice

def graph(X):
	g = EventGraph()
	for aid, seq in enumerate (X):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom ))
	g.buildEdges(delta = 1)
	print nx.info(g)
	return g

def sample2():
	g = EventGraph()
	left = norm(loc=np.array([5.0]), scale=1)
	right = norm(loc=np.array([-5.0]), scale=1)
	stop = norm(loc=np.array([0.0]), scale=1)
	#sample = np.random.choice([left, right, stop, None], size=(4,6), p=[0.3,0.3,0.3,0.1])
	#sample = choice([left, right, stop], size=(1,30), p=[0.4,0.4,0.2])
	sample = [[left, left, left, left, stop, stop, right, right, right]*4]
	rvs = np.frompyfunc(lambda x: x.rvs(), 1, 1)
	samples = rvs(sample)
	return samples, None, None

def sample():
	###############################################################################
	# Downloading the data
	date1 = datetime.date(2012, 1, 1)  # start date
	date2 = datetime.date(2012, 12, 1)  # end date
	# get quotes from yahoo finance
	quotes = quotes_historical_yahoo("INTC", date1, date2)
	if len(quotes) == 0:
	    raise SystemExit

	# unpack quotes
	dates = np.array([q[0] for q in quotes], dtype=int)
	close_v = np.array([q[2] for q in quotes])
	#volume = np.array([q[5] for q in quotes])[1:]

	# take diff of close value
	# this makes len(diff) = len(close_t) - 1
	# therefore, others quantity also need to be shifted
	diff = 100 * ( np.exp( np.log(close_v[1:]) - np.log(close_v[:-1]) ) - 1 )
	dates = dates[1:]
	close_v = close_v[1:]
	print diff
	# pack diff and volume for training
	#X = np.column_stack([diff, volume])
	X = np.column_stack([diff])
	return X, dates, close_v

def train(X, n_components):
	###############################################################################
	# Run Gaussian HMM
	print("fitting to HMM and decoding ...")
	

	# make an HMM instance and execute fit
	model = GaussianHMM(
		n_components, 
		covariance_type="diag", 
		n_iter=2000)

	model.fit([X])

	# predict the optimal sequence of internal hidden state
	hidden_states = model.predict(X)

	print("done\n")

	###############################################################################
	# print trained parameters and plot
	print("Transition matrix")
	print(model.transmat_)
	print()

	print("means and vars of each hidden state")
	for i in range(n_components):
	    print("%dth hidden state" % i)
	    print("mean = ", model.means_[i])
	    print("var = ", np.diag(model.covars_[i]))
	    print()

	return hidden_states

def vis(dates, close_v, n_components):
	years = YearLocator()   # every year
	months = MonthLocator()  # every month
	yearsFmt = DateFormatter('%Y')
	fig = pl.figure()
	ax = fig.add_subplot(111)

	for i in range(n_components):
	    # use fancy indexing to plot data in each state
	    idx = (hidden_states == i)
	    ax.plot_date(dates[idx], close_v[idx], 'o', label="%dth hidden state" % i)
	ax.legend()

	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	ax.xaxis.set_minor_locator(months)
	ax.autoscale_view()

	# format the coords message box
	ax.fmt_xdata = DateFormatter('%Y-%m-%d')
	ax.fmt_ydata = lambda x: '$%1.2f' % x
	ax.grid(True)

	fig.autofmt_xdate()
	pl.show()

def vis2(dates, close_v, graph):
	years = YearLocator()   # every year
	months = MonthLocator()  # every month
	yearsFmt = DateFormatter('%Y')
	fig = pl.figure()
	ax = fig.add_subplot(111)

	idx1 = [n._tp for (n,d) in graph.nodes(data=True) 
		if d.get('cluster', 'white') == "pink"]
	idx2 = [n._tp for (n,d) in graph.nodes(data=True) 
		if d.get('cluster', 'white') == "white"]

	
	ax.plot_date(dates[idx1], close_v[idx1], 'o', color='pink')
	ax.plot_date(dates[idx2], close_v[idx2], 'o', color='blue')
	ax.plot_date(dates, close_v, '--')
	#for n in nodes1 :
	#	ax.scatter(dates[n._tp], n._semantics, 'o', color='pink')
	#for n in nodes2 :
	#	ax.scatter(dates[n._tp], n._semantics, 'o', color='blue')

	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	ax.xaxis.set_minor_locator(months)
	ax.autoscale_view()

	# format the coords message box
	ax.fmt_xdata = DateFormatter('%Y-%m-%d')
	ax.fmt_ydata = lambda x: '$%1.2f' % x
	ax.grid(True)

	fig.autofmt_xdate()
	pl.show()

def drawbox(ax, graph):
	#positions = [ x._tp for x, y in graph.edges()]
	#medians = [ x._semantics[0] for x, y in graph.edges()]
	#sm = semanticMatrix(graph)
	#ax.boxplot(sm.T, positions=positions, usermedians=medians)
	for x in graph.nodes():
		yy = filter(lambda x: x!=0, x._semantics)
		if len(yy) > 0 :
			xx = [x._tp] * len(yy)
			print xx
			print yy
			ax.plot(xx, yy, '_-', color='red', markersize=15)

def drawG3(ax, graph, samples):
	# only for vis 1 dimensonal data 
	drawbox(ax, graph)
	ax.plot(samples[0], '--.', color='black',alpha=0.3)
	ax.grid()

if __name__ == '__main__':
	np.set_printoptions(precision=2)
	n_components = 3
	X, dates, close_v = sample2()
	#hidden_states = train(X, n_components)
	#vis(dates, close_v, n_components)
	print X
	g = graph(X)
	gamma = -5
	figsize=(18,8)
	fig = pl.figure(figsize=figsize)
	N = 6
	for i in range (N) :
		ax = fig.add_subplot(N, 1, i+1)
		ax.set_title(i+1)
		drawG3(ax, g, X)
		try:
			means, covars = toProd(g, k=10, gamma=gamma)
		
			print means.reshape(2, len(means)/2)
			print covars.reshape(2, len(covars)/2)
			means, covars =  cutDim(means, covars)
		#pprint (g.nodes(data=True))
		#drawG2(g, node_size=2000, cluster=True, label=True, output="pics/test_%s"%i, 
		#		title="%s"%(means.reshape(2, len(means)/2)))
		
			rewrite(g, means, covars, gamma=gamma)
			
		except:
			plt.show()
	plt.show()
	