import numpy as np
from sklearn.hmm import GaussianHMM

class Triangle(GaussianHMM):
  def __init__(self, *args, **kwargs):
    super(GaussianHMM, self).__init__(*args, **kwargs)
    self.name = "triangle"
    self._covariance_type = "diag"
    self.means_ = np.array([[1e-2], [0], [-1e-2]])
    self.covars_ = np.array([[1e-4], [1e-4], [1e-4]])

if __name__ == "__main__":
  startprob = np.array([.4, .2, .4])
  transmat = np.array([[.8, .1, .1], [.4, .2, .4], [.1, .1, .8]])
  model = Triangle(3, startprob, transmat)
  X,Z = model.sample(40) 
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.plot(X, "-o")
  ax2.plot(Z, "-o")
  plt.show()

