library("depmixS4")
library("quantmod")
getSymbols("uup", src="yahoo")
Nsma <- 7
price <- UUP["2012-10-15/2012-12-20"][, 4]
price <- SMA(price, Nsma)[(Nsma+1):length(price),]
ret <- data.frame(ret=100 * diff(log(price)))
colnames(ret) <- "logret"
set.seed(1)
mod <- depmix(logret ~ 1, data = ret, nstates = 3)
fm <- fit(mod, emc=em.control(rand=TRUE))
summary(fm, which="transition")
par(mfrow=c(2,1))
plot(price, main="sma(10)")
plot(100*diff(log(price)), type='p')
resp1 <- fm@response[[1]][[1]]@parameters$coefficients
resp2 <- fm@response[[2]][[1]]@parameters$coefficients
resp3 <- fm@response[[3]][[1]]@parameters$coefficients
abline(h=resp1, col="red")
abline(h=resp2, col="red")
abline(h=resp3, col="red")