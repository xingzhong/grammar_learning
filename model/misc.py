import numpy as np
from matplotlib.finance import quotes_historical_yahoo
import matplotlib.pyplot as plt
import datetime

N = 32 

start = datetime.date(2011,1,1)
end = datetime.date(2013,2,1)
quotes = quotes_historical_yahoo("uup", start, end)
dates = np.array([q[0] for q in quotes], dtype=int)
close = np.array([q[2] for q in quotes])
logClose = np.log(close)
logRet = np.diff([logClose]).T
dates = dates[1:]
close = close[1:]
xvals = np.linspace(dates[0],dates[-1],N)
close_inter = np.interp(xvals, dates, np.diff(logClose))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot_date(dates, close)
ax2.plot_date(xvals, close_inter, '-')

ax1.autoscale_view()
fig.autofmt_xdate()
plt.show()
