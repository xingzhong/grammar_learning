import numpy as np
from sklearn import hmm
from itertools import permutations

def logLik(obs, transmat):
  ll = 0.0
  pairs = zip(obs, obs[1:])
  for x,y in pairs:
    ll += np.log(transmat[x,y])
  return ll

symbols = dict(zip(range(3), ['a','b', 'c']))
startprob = np.array([1.0, 0, 0])
transmat = np.array([[0.2, 0.8, 0.0], [1/3.0, 1/3.0, 1/3.0], [1.0, 0, 0]])
emissionprob = np.eye(3)
model = hmm.MultinomialHMM(n_components=3,
              startprob = startprob,
              transmat = transmat)
model.emissionprob_ = emissionprob
obs, state = model.sample(n=14)

print state
outputs = map(lambda x:symbols[x], state)
print " ".join(outputs)

ssample = model.score(obs)
print ssample
#standard = [0,1,0,1,1,2,0,0,1,0,1,1,2,0]
#print standard
#print model.score(standard)

for _ in range(30):
  obs, s = model.sample(n=14)
  
  score = model.score(s)
  if abs(score-ssample) < 0.2 :
    print 'found'
    #print s
    outputs = map(lambda x:symbols[x], s)
    print " ".join(outputs)
    print score

