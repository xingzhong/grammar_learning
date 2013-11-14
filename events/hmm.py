from event import *
from test import toProd, rewrite
import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.hmm import GaussianHMM

def graph(X):
	g = EventGraph()
	for aid, seq in enumerate (X):
		for t, atom in enumerate (seq):
			if not atom is None:
				g.addEvent( Event(t, aid, atom ))
	g.buildEdges(delta = 1)
	print nx.info(g)
	return g


def sample():
	###############################################################################
	# Downloading the data
	date1 = datetime.date(2011, 01, 1)  # start date
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

if __name__ == '__main__':
	np.set_printoptions(precision=2)
	n_components = 3
	X, dates, close_v = sample()
	#hidden_states = train(X, n_components)
	#vis(dates, close_v, n_components)
	#print X
	g = graph([X])
	
	for i in range (4) :
		if len(g.nodes()) < 2 :
			break
		means, covars = toProd(g, k=15, gamma=-5)
		
		print means.reshape(2, len(means)/2)
		print covars.reshape(2, len(covars)/2)
		

		#drawG2(g, node_size=2000, cluster=True, label=True, output="pics/test_%s"%i, 
		#		title="%s"%(means.reshape(2, len(means)/2)))

		rewrite(g, means, covars, gamma=-5)