library("depmixS4")

model1 <- function( data, response=logret~1 ){
  # given data, return the likeihood
  ret <- data.frame(logret = data)
  mod <- depmix(
    response = response,
    data = ret,
    nstates = 3,
    instart = c (0.8, 0.1, 0.1),    # init state P
    trstart = c (0.92, 0.08, 0,     # transtation P matrix
                 0.03, 0.87, 0.10,
                 0, 0.08, 0.92),
    respstart = c(0.2, 0.1,     # resoponse model
                  -0, 0.1,
                  -0.2, 0.1)
  )
  return (logLik(mod))
}

model2 <- function( data, response=logret~1 ){
  # given data, return the likeihood
  ret <- data.frame(logret = data)
  mod <- depmix(
    response = response,
    data = ret,
    nstates = 3,
    instart = c (1, 0, 0),    # init state P
    trstart = c (1, 0, 0,     # transtation P matrix
                 1, 0, 0,
                 1, 0, 0),
    respstart = c(0.0, 0.3,     # resoponse model
                  0.0, 0.1,
                  0.0, 0.1)
  )
  return (logLik(mod))
}

model3 <- function( data, response=logret~1 ){
  # given data, return the likeihood
  ret <- data.frame(logret = data)
  mod <- depmix(
    response = response,
    data = ret,
    nstates = 3,
    instart = c (0.3, 0.3, 0.3),    # init state P
    trstart = c (0.3, 0.3, 0.3,     # transtation P matrix
                 0.3, 0.3, 0.3,
                 0.3, 0.3, 0.3),
    respstart = c(0.2, 0.1,     # resoponse model
                  0.0, 0.1,
                  -0.2, 0.1)
  )
  return (logLik(mod))
}