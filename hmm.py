import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
from sklearn import hmm

def debug(x):
  print "[debug]", x
  return x
# TODO:  
#   down-sampling
#   partial tranining 
startprob1 = np.array([0.4, 0.2, 0.4])
transmat1 = np.array([[0.8, 0.1, 0.1], [0.4, 0.2, 0.4], [0.1, 0.1, 0.8]])
means = np.array([[0.5*1e-2], [1e-6], [-0.5*1e-2]])
covars = np.array([[1e-4], [1e-4], [1e-4]])

model1 = hmm.GaussianHMM(3, 'diag', startprob1, transmat1)
model1.means_ = means
model1.covars_ = covars

startprob2 = np.array([0.33, 0.33, 0.34])
transmat2 = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])

model2 = hmm.GaussianHMM(3, 'diag', startprob2, transmat2)
model2.means_ = means
model2.covars_ = covars

start = datetime.date(2012,1,1)
end = datetime.date(2013,1,1)
quotes = quotes_historical_yahoo("uup", start, end)
dates = np.array([q[0] for q in quotes], dtype=int)
close = np.array([q[2] for q in quotes])
logClose = np.log(close)
logRet = np.diff(logClose)
dates = dates[1:]
close = close[1:]

N = 32 #windows size
score1 = []
score2 = []
for i in range(len(logRet)):
  print "%s/%s"%(i, len(logRet))
  for j in range(i+16, min(len(logRet), i+5*16)):
    w = logRet[i:j]
    xvals = np.linspace(dates[i], dates[j], N)
    w_interp = np.interp(xvals, dates[i:j], w)
    w_interp = np.array([w_interp]).T
    score1.append([model1.score(w_interp), i, j])
    score2.append([model2.score(w_interp), i, j])

w1 = sorted(score1, key=lambda x: x[0])[-1:]
w2 = sorted(score2, key=lambda x: x[0])[-1:]
debug(w1)
debug(w2)
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot_date(dates, close, "-o")
bx = fig.add_subplot(212)
bx.plot_date(dates, logRet, "-o")
for ww in w1:
  ax.axvspan(dates[ww[1]], dates[ww[2]], facecolor='g', alpha=0.5)
  bx.axvspan(dates[ww[1]], dates[ww[2]], facecolor='g', alpha=0.5)
for ww in w2:
  ax.axvspan(dates[ww[1]], dates[ww[2]], facecolor='r', alpha=0.5)
  bx.axvspan(dates[ww[1]], dates[ww[2]], facecolor='r', alpha=0.5)
ax.autoscale_view()
fig.autofmt_xdate()
plt.show()
