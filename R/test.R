library("depmixS4")
library("quantmod")
library("zoo")
getSymbols("uup", src="yahoo")
Nsma <- 7
price <- SMA( UUP["2012-06-09/2012-09-01"][, 4], Nsma)
price <- price[!is.na(price[,1])]
logret <- 100*diff(log(price))[-1,]
ret <- data.frame(logret = logret)
# logret is the log return of the given series

set.seed(2)
mod <- depmix(
  response = logret~1,
  data = ret,
  nstates = 3,
  instart = c (0.8, 0.1, 0.1),    # init state P
  trstart = c (0.92, 0.08, 0,     # transtation P matrix
               0.03, 0.87, 0.10,
               0, 0.08, 0.92),
  respstart = c(0.178, 0.062,     # resoponse model
                -0.0139, 0.049,
                -0.172, 0.072)
)

fb <- forwardbackward(mod)
print(fb)
par(mfrow=c(2,1)) 
plot(price)
plot(log(fb$sca), type="l")

# likihood are calculated through the summation of the scale factor. 
# Therefore, by finding the minimum peroid of scale factors, 
# we could achive a good likihood
# the scale factor is determined by the summation of the entire states response 
# probability, if we could guarantee the repsonse have high probability locate 
# in one paticular state, the scale factor should be smaller.
