import numpy as np
import datetime
from matplotlib.finance import quotes_historical_yahoo
import matplotlib.pyplot as plt
from sklearn import hmm

start = datetime.date(2012,4,27)
end = datetime.date(2012,9,14)
quotes = quotes_historical_yahoo("uup", start, end)
dates = np.array([q[0] for q in quotes], dtype=int)
close = np.array([q[2] for q in quotes])
logClose = np.log(close)
logRet = np.diff(logClose)
dates = dates[1:]
close = close[1:]

n_components = 3
N = 16
model = hmm.GaussianHMM(n_components, "diag")
xvals = np.linspace(dates[1], dates[-1], N)
logRetInterp = np.interp(xvals, dates, logRet)
model.fit( [np.array([logRetInterp]).T]  )

#####################################################
# print trained parameters and plot
print "Transition matrix"
print model.transmat_
print ""

print "means and vars of each hidden state"
for i in xrange(n_components):
  print "%dth hidden state" % i
  print "mean = ", model.means_[i]
  print "var = ", np.diag(model.covars_[i])
  print ""

fig = plt.figure()
ax = fig.add_subplot(411)
ax.plot_date(dates, close, "-o")
ax.grid(True)
bx = fig.add_subplot(412)
bx.plot_date(dates, logRet, "-o")
bx.grid(True)
cx = fig.add_subplot(413)
cx.plot_date(xvals, logRetInterp, "-o")
cx.grid(True)
dx = fig.add_subplot(414)
dx.plot_date(dates, model.sample(len(dates))[0], "-o")
dx.grid(True)
ax.autoscale_view()
fig.autofmt_xdate()
plt.show()
