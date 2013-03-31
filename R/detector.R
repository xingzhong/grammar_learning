library("quantmod")
library("zoo")
getSymbols("uup", src="yahoo")
Nsma <- 7
price <- SMA( UUP["2012-01-01/2012-04-01"][, 4], Nsma)
price <- price[!is.na(price[,1])]
logret <- 100*diff(log(price))[-1,]
# logret is the log return of the given series

source("./eval.R")
test <- rollapply(logret, width = 20, 
                  by = 1,
                  align = "left",
                  FUN = evalModel,
                  response = logret~1,
                  by.column = TRUE
                  )



