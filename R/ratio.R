source("./model.R")
library("quantmod")
getSymbols("uup", src="yahoo")
Nsma <- 7
price <- SMA( UUP["2011-01-01/2012-10-01"][, 4], Nsma)
price <- zoo(price[!is.na(price[,1])])
logret <- 100*diff(log(price))[-1,]

lik1 <- rollapply(logret, width = 30, 
                  by = 1,
                  align = "left",
                  FUN = model1,
                  response = logret~1)
lik2 <- rollapply(logret, width = 30, 
                  by = 1,
                  align = "left",
                  FUN = model2,
                  response = logret~1)
lik3 <- rollapply(logret, width = 30, 
                  by = 1,
                  align = "left",
                  FUN = model3,
                  response = logret~1)
plot( merge(price, lik1, lik2, lik3), screens=c(1,2,2,2), col=c('black','blue','red','green') )
grid()