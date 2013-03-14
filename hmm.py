import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
from sklearn import hmm

def debug(x):
  print "[debug]", x
  return x

startprob1 = np.array([0.4, 0.2, 0.4])
transmat1 = np.array([[0.8, 0.1, 0.1], [0.4, 0.2, 0.4], [0.1, 0.1, 0.8]])
means1 = np.array([[1], [1e-5], [1]])
covars1 = np.array([[1.0], [1.0], [1.0]])

model1 = hmm.GaussianHMM(3, 'diag', startprob1, transmat1)
model1.means_ = means1
model1.covars_ = covars1

startprob2 = np.array([0.33, 0.33, 0.34])
transmat2 = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
means2 = np.array([[1], [1e-5], [-1]])
covars2 = np.array([[1.0], [1.0], [1.0]])

model2 = hmm.GaussianHMM(3, 'diag', startprob2, transmat2)
model2.means_ = means2
model2.covars_ = covars2

#X, Z = model.sample(80)
#plt.plot(np.cumsum(X), "-o", mfc="orange")

start = datetime.date(2011,1,1)
end = datetime.date(2013,2,1)
quotes = quotes_historical_yahoo("SPY", start, end)
dates = np.array([q[0] for q in quotes], dtype=int)
close = np.array([q[2] for q in quotes])
diff = np.diff([close]).T
dates = dates[1:]
close = close[1:]

print len(diff)
N = 32 #windows size
score1 = []
score2 = []
for i in range(len(diff)-N):
  w = diff[i:i+N]
  score1.append([model1.score(w), i])
  score2.append([model2.score(w), i])

w1 = sorted(score1, key=lambda x: x[0])[-2:]
w2 = sorted(score2, key=lambda x: x[0])[-2:]
debug(w1)
debug(w2)
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot_date(dates, close, "-o")
bx = fig.add_subplot(212)
bx.plot_date(dates, diff, "-o")
for ww in w1:
  ax.axvspan(dates[ww[1]], dates[ww[1]+N], facecolor='g', alpha=0.5)
  bx.axvspan(dates[ww[1]], dates[ww[1]+N], facecolor='g', alpha=0.5)
for ww in w2:
  ax.axvspan(dates[ww[1]], dates[ww[1]+N], facecolor='r', alpha=0.5)
  bx.axvspan(dates[ww[1]], dates[ww[1]+N], facecolor='r', alpha=0.5)
ax.autoscale_view()
fig.autofmt_xdate()
plt.show()
