import pandas as pd
import numpy as np
from pandas.io.data import DataReader
import datetime
from sklearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

def draw(df):
	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(111)
	bx = ax.twinx()
	ax.plot(df.index, df['Adj Close'], '.')
	ax.plot(df.index, df['predict'], 'r--')
	ax.set_ylim([0, 1.1*df['Adj Close'].max()])
	
	ax.grid()
	bx.bar(df.index, df['Volume'], color='g', alpha=0.4, log=True)
	bx.set_ylim([df.Volume.min(), 100*df.Volume.max()])
	
	plt.show()

def filtering(df, model):
    # perform filtering on pandas dataframe
    # dataframe is time indexed from prior to present 
    # in filtering, each prediction is based on histroical data
    # ideally the histrocial states are Markov relationship
    # so we only need to recursive on 1st order 
    # 
    # in dataframe, at least we will have following cols,
    # time index, underlying series, hidden states, predict states
    # 
    # an other param is statistical model,
    # take the hidden Markov model as an example, 
    # each sample in underlying series is one of our observation,
    # and hidden states will modeled as part of a Markov chain,
    # so for given hidden state, we can predicate next hidden state 
    # based upon the trainsition matrix, 
    # and given the current observation, we can update the current 
    # state liklihood based on emission probability
    #
    # initally we use simple for-loop implementation, 
    # then we use more advanced functional style implemetation.
    obs = np.asarray(df.values)
    m, _ = obs.shape
    n, _ = model.means_.shape
    framelogprob = model._compute_log_likelihood(obs)
    
    df = df.join( pd.DataFrame(
    	framelogprob, index=df.index, columns=map(lambda x: "frame_"+str(x), range(n)) ))
    _, fwdlattice = model._do_forward_pass(framelogprob)
    df = df.join( pd.DataFrame(
    	fwdlattice, index=df.index, columns=map(lambda x: "forward_"+str(x), range(n)) ))
    df['hidden'] = np.argmax(fwdlattice, axis=1)
    _, df['viterbi'] = model.decode(obs)
    df['predict'] = map(lambda x: model.means_[x, 0], df.hidden)
    draw(df)
    #import pdb; pdb.set_trace()
    for row in df.iterrows() :
        pass

    print df.head()

def hmm(samples):
	model = GaussianHMM(n_components=3)
	samples = samples.dropna()
	idx = samples.index
	if samples.values.ndim < 2:
		#import pdb; pdb.set_trace()
		m = samples.values.shape
		samples = samples.values.reshape(m[0],1)
	
	model.fit([samples])
	#_, states = model.decode(samples, algorithm='map')
	framelogprob = model._compute_log_likelihood(samples)
	logprob, fwdlattice = model._do_forward_pass(framelogprob)
	
	n, _ = model.means_.shape
	frame = pd.DataFrame(
    	framelogprob, index=idx, columns=map(lambda x: "frame_"+str(x), range(n)) )
	forward = pd.DataFrame(
    	fwdlattice, index=idx, columns=map(lambda x: "forward_"+str(x), range(n)) )
	#import pdb; pdb.set_trace()
	predict = pd.DataFrame(
		(fwdlattice-framelogprob)[1:, :], index=idx[:-1], columns=map(lambda x: "predict_"+str(x), range(n)))
	import pdb; pdb.set_trace()
	return model, frame.join(forward)

def predicate(states, models):
	# given current state
	# return mean of next observation
	pass

if __name__ == '__main__':
	start = datetime.datetime(2012, 1, 1)
	end = datetime.datetime(2013, 1, 27)
	df = DataReader("F", 'yahoo', start, end)
	sample = df['Adj Close'].diff()
	model, fwd = hmm(sample)
	
	import pdb; pdb.set_trace()

	#filtering(df, model)