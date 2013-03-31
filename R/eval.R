library("depmixS4")

evalModel <- function( data, response=logret~1 ){
  # given data, return the likeihood
  print ("evalModel")
  ret <- data.frame(logret = data)
  mod <- depmix(
    response = response,
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
  scale <- -log(fb$sca)
  #scale <- as.xts(-log(fb$sca), order.by=index(data))
  
  ind <- which(scale==min(scale))
  print (index(data)[1])
  print (index(data)[ind])
  ret <- sprintf("%s/%s", index(data)[1], index(data)[ind])
  print (ret)
  return (ret)
}