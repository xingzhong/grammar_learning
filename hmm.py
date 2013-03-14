import numpy as np
from sklearn import hmm

startprob = np.array([1.0, 0.0, 0.0])
transmat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
means = np.array([[1.0], [0.0], [-1.0]])
covars = np.array([[1.0], [1.0], [1.0]])

model = hmm.GaussianHMM(3, 'diag', startprob, transmat)
model.means_ = means
model.covars_ = covars

X, Z = model.sample(20)

