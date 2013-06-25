from pprint import pprint 
import numpy as np
from production import *

class Tree:
  def __init__(self, arg):
    self.symbol = arg
    self.children = []

  def __repr__(self):
    return "%s->%s"%(str(self.symbol), str(self.children))

  def addNode(self, node):
    self.children.append( node )

  def walk(self):
    print '[', self.symbol,
    for n in self.children:
      n.walk()
    print ']',


production = Productions()
production[(NT('S'), (None, NT('A'), None))] = 0.33
production[(NT('S'), (None, NT('B'), None))] = 0.33
production[(NT('S'), (None, NT('C'), None))] = 0.33
production[(NT('A'), (T('a'), NT('A'), T('a')))] = 0.6
production[(NT('A'), (T('a'), NT('B'), T('a')))] = 0.3
production[(NT('A'), (None,))] = 0.1
production[(NT('B'), (T('b'), NT('A'), None))] = 0.33
production[(NT('B'), (T('b'), NT('B'), None))] = 0.33
production[(NT('B'), (T('b'), NT('C'), None))] = 0.33
production[(NT('B'), (None,))] = 0.01
production[(NT('C'), (None, NT('B'), T('c')))] = 0.6
production[(NT('C'), (None, NT('C'), T('c')))] = 0.3
production[(NT('C'), (None,))] = 0.1


x = "b a b a c"
x = x.split()
L = len(x)
pprint (x)

gamma = {}
path = {}
# init 
for i in range(2, L+1):
  gamma[(i, i-1)] = np.log(rule[('-')])
  path[(i,i-1)] = '*'

# iter
for dj in range(L):
  for i in range(1, L+1):
    if i+dj <= L:
      j = i+dj
      print (i,j)
      temp = []
      temp2 = []
      if i+1 <= L and j-1 > 0 and i!=j:
        print "\tg(%s, %s)+logP(S -> %s S %s)"%(i+1, j-1, x[i-1], x[j-1])
        #s = Tree(('S',i,j), (x[i-1],i), ('S',i+1, j-1), (x[j-1],j))
        s = ( x[i-1], (i+1, j-1), x[j-1] )
        t = path.get((i+1, j-1),'*')
        prods = production.match(t1=x[i-1], t2=x[j-1], nt=t[0])
        for p in prods:
          if gamma.has_key((i+1, j-1)):
            temp2.append( (gamma[(i+1, j-1)] + np.log(production[p]), p) )
        #temp.append( (gamma[(i+1, j-1)] + np.log(rule.get((x[i-1], 'S', x[j-1]), 0)), s ))
      if i+1 <= L :
        print "\tg(%s, %s)+logP(S -> %s S)"%(i+1, j, x[i-1])
        #s = Tree(('S',i,j), (x[i-1],i), ('S',i+1, j))
        t = path.get((i+1, j),'*')
        prods = production.match(t1=x[i-1], nt=t[0])
        for p in prods:
          if gamma.has_key((i+1, j)):
            temp2.append( (gamma[(i+1, j)] + np.log(production[p]), p) )
        s = ( x[i-1], (i+1, j) )
        #temp.append( (gamma[(i+1, j)] + np.log(rule.get((x[i-1], 'S'), 0)), s))
      if j-1 > 0 :
        print "\tg(%s, %s)+logP(S -> S %s)"%(i, j-1, x[j-1])
        #s = Tree(('S',i,j), ('S',i, j-1), (x[j-1], j))
        s = ((i, j-1), x[j-1])
        t = path.get((i, j-1),'*')
        prods = production.match(t2=x[j-1], nt=t[0])
        for p in prods:
          if gamma.has_key((i, j-1)):
            temp2.append( (gamma[(i, j-1)] + np.log(production[p]), p) )
        #temp.append( (gamma[(i, j-1)] + np.log(rule.get(('S', x[j-1]), 0)), s))
      for k in range(i+1, j):
        print "\t\tg(%s,%s)+g(%s, %s)"%(i,k,k+1,j)
      #tmax, p = max(temp)
      pprint(temp2)
      if len(temp2) > 0:
        tmax, p = max(temp2)
        gamma[(i,j)] = tmax
        path[(i,j)] = p

pprint(gamma)
pprint(path)
pprint(production)


def build(node):
  #import pdb; pdb.set_trace()
  if not isinstance(node, tuple):
    return Tree(node)
  tree = Tree(node)
  children = path.get(node, node)
  if children:
    tree.children.extend(map(build, children))
  return tree

print "build tree"
start = (1,L)
tree = build(start)
tree.walk()
print 
print gamma[start]
