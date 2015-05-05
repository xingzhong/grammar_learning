import cv2
import numpy as np 
import pandas as pd 
import cPickle as pickle
from matplotlib import pyplot as plt
from template import template
from patsy import dmatrices
import statsmodels.api as sm
np.set_printoptions(suppress=True)


def main():
	ms = pickle.load( open("sep/25_68351.ms", "rb") )
	mtxRs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[1], ms))
	mtxQs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[2], ms))
	n, _, _ = mtxRs.shape
	params = []
	for i in range(n):
		r = mtxRs[i]
		q = mtxQs[i]
		theta = -np.arcsin(q[0,1])
		dx = r[0,2]
		dy = r[1,2]
		focus = r[1,1]
		shear = r[0,1]
		ratio = r[0,0] / focus
		params.append([dx, dy, focus, theta, shear, ratio, q[2,0], q[2,1]])
	df = pd.DataFrame(params, columns=['dx', 'dy',  'focus', 'theta',  'shear', 'ratio', 'd1', 'd2'])
	theta, X = dmatrices('theta ~ dx + dy + focus', data=df, return_type='dataframe')
	modTheta = sm.OLS(theta, X)
	res = modTheta.fit() 
	print res.summary()
	shear, X = dmatrices('shear ~ dx + dy + focus', data=df, return_type='dataframe')
	modshear = sm.OLS(shear, X)
	res = modshear.fit() 
	print res.summary()
	ratio, X = dmatrices('ratio ~ dx + dy + focus', data=df, return_type='dataframe')
	modratio = sm.OLS(ratio, X)
	res = modratio.fit() 
	print res.summary()
	d1, X = dmatrices('d1 ~ dx + dy + focus', data=df, return_type='dataframe')
	modd1 = sm.OLS(d1, X)
	res = modd1.fit() 
	print res.summary()
	d2, X = dmatrices('d2 ~ dx + dy + focus', data=df, return_type='dataframe')
	modd2 = sm.OLS(d2, X)
	res = modd2.fit() 
	print res.summary()
	sm.graphics.plot_partregress('theta', 'dx', ['focus', 'dy'], data=df, obs_labels=False)
	sm.graphics.plot_partregress('shear', 'dx', ['focus', 'dy'], data=df, obs_labels=False)
	sm.graphics.plot_partregress('ratio', 'dx', ['focus', 'dy'], data=df, obs_labels=False)
	sm.graphics.plot_partregress('d1', 'dx', ['focus', 'dy'], data=df, obs_labels=False)
	sm.graphics.plot_partregress('d2', 'dx', ['focus', 'dy'], data=df, obs_labels=False)
	plt.show()
	#import ipdb; ipdb.set_trace()
	#df.plot(subplots=True, layout=(3, 3), sharex=True, title='Example of the parameter evolution through time')
	#plt.show()

if __name__ == '__main__':
	main()