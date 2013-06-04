from pprint import pprint 
import numpy as np

# S -> a S a [0.33] | b S [0.33] | S c [0.33] | - [0.01]
rule = {}
rule[('a', 'S', 'a')] = 0.33
rule[('b', 'S')] = 0.33
rule[('S', 'c')] = 0.33
rule[('-')] = 0.01

L = 5
x = "a b b a c"
x = x.split()
pprint (x)

gamma = {}
path = []
# init 
for i in range(2, L+1):
  gamma[(i, i-1)] = np.log(rule[('-')])

# iter
for dj in range(L):
  for i in range(1, L+1):
    if i+dj <= L:
      j = i+dj
      print (i,j)
      temp = []
      if i+1 <= L and j-1 > 0 and i!=j:
        print "\tg(%s, %s)+logP(S -> %s S %s)"%(i+1, j-1, x[i-1], x[j-1])
        s = "S -> %s S %s"%(x[i-1], x[j-1])
        temp.append( (gamma[(i+1, j-1)] + np.log(rule.get((x[i-1], 'S', x[j-1]), 0)), s ))
      if i+1 <= L :
        print "\tg(%s, %s)+logP(S -> %s S)"%(i+1, j, x[i-1])
        s = "S -> %s S"%(x[i-1])
        temp.append( (gamma[(i+1, j)] + np.log(rule.get((x[j-1], 'S'), 0)), s))
      if j-1 > 0 :
        print "\tg(%s, %s)+logP(S -> S %s)"%(i, j-1, x[j-1])
        s = "S -> S %s"%(x[j-1])
        temp.append( (gamma[(i, j-1)] + np.log(rule.get(('S', x[j-1]), 0)), s))
      for k in range(i+1, j):
        print "\t\tg(%s,%s)+g(%s, %s)"%(i,k,k+1,j)
      tmax, p = max(temp)
      gamma[(i,j)] = tmax
      pprint(p)

pprint(path)
pprint(gamma)
