# State : two branch 
from scipy.stats import norm
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import numpy as np
import sys
from pprint import pprint 

class State(object):
  def __init__(self, id, left=None, right=None):
    self._id = id               # name
    if left:
      self._left = Dist(left)     # left pdf
    else:
      self._left = None
    if right:
      self._right = Dist(right)   # right pdf
    else:
      self._right = None
  def __repr__(self):
    return self._id
  def left(self):
    return self._left
  def right(self):
    return self._right

class Dist(object):
  def __init__(self, args):
    self._mu = np.matrix(args[0])
    self._sigma = np.matrix(args[1])
  def __repr__(self):
    return "normal(%s, %s)"%(self._mu, self._sigma)

  def logpdf(self, x):
    S = self._sigma
    mu = self._mu
    nx = len(S)
    tmp = -0.5*(nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1])
    x = np.matrix(x)
    err = x.T-mu.T
    if (sp.issparse(S)):
      numerator = spln.spsolve(S, err).T.dot(err)
    else:
      numerator = np.linalg.solve(S, err).T.dot(err)
    return tmp-numerator

  def pdf(self, x):
    return math.exp(self.logpdf(x))


def draw(obs, states, fig):
  for ind, o in enumerate(obs):
    fig.add_node(ind, label="%.3f"%o, pos="%s,%s!"%(2*ind, 1))
  for ind, s in enumerate(states):
    for t in range(len(obs)):
      fig.add_node((s, t), pos="%s,%s!"%(2*t, -ind), label="{%s|%s}"%(s[0], s[2]))
  return fig

def isPairs(path):
  stack = []
  for s in path:
    print s
    if s[4] == "double":
      if s[2] == "left":
        stack.append(s[0])
      if s[2] == "right":
        ss = stack.pop()
        print ss
  print stack

def viterbi(obs, states, trans_p, fig):
  V = [{}]
  path = {}

  for y in states:
    V[0][y] = y[3] * y[1].pdf(obs[0])
    path[y] = [y]
    node = fig.get_node((y,0))
    node.attr['label'] = "%s|%.3f"%(node.attr['label'], np.log(V[0][y]))

  for t in range(1, len(obs)):
    V.append({})
    newpath = {}
    for y in states:
      (prob, state) = max([(V[t-1][y0]*trans_p.get((y0[0],y[0]), 0)*y[1].pdf(obs[t]), y0) for y0 in states])
      V[t][y] = prob
      node = fig.get_node((y,t))
      node.attr['label'] = "%s|%.3f"%(node.attr['label'], np.log(prob))
      fig.add_edge((state, t-1), (y,t))
      print ( (state[0], y[0]))
      newpath[y] = path[state] + [y]
      isPairs(newpath[y])
    path = newpath
  pprint (V)
  (prob, state) = max( [(V[len(obs)-1][y], y) for y in states] )
  return (prob, path[state])

obs = [0.0, 1.0, -1.0, -1.0, 1.0, 0.0]
A1 = ("A", Dist((0.0, 1.0)), "left", 1.0/3, "double")
A2 = ("A", Dist((0.0, 1.0)), "right", 0.0, "double")
B  = ("B", Dist((1.0, 1.0)), "left", 1.0/3, "single")
C  = ("C", Dist((-1.0, 1.0)), "left", 1.0/3, "single")
states = [A1, A2, B, C]
trans = {}
trans[("A","A")] = 1.0/2 
trans[("B","B")] = 1.0/2 
trans[("C","C")] = 1.0/2 
trans[("A","B")] = 1.0/2
trans[("B","A")] = 1.0/2
trans[("B","C")] = 1.0/2
trans[("C","B")] = 1.0/2

import pygraphviz as pgv
fig = pgv.AGraph(rankdir="LR", splines="line")
fig.node_attr["shape"] = "record"
fig = draw(obs, states, fig)

pprint( viterbi(obs, states, trans, fig) )

fig.layout(prog='neato')
fig.draw("test.png")
sys.exit(0)


A = State("A", (0.0, 1.0), (0.0, 1.0))
B = State("B", (1.0, 1.0))
C = State("C", (-1.0, 1.0))

states = [A, B, C]
trans = {}
inits = {}
trans[(A,A)] = 1.0/2 
trans[(B,B)] = 1.0/2 
trans[(C,C)] = 1.0/2 
trans[(A,B)] = 1.0/2
trans[(B,A)] = 1.0/2
trans[(B,C)] = 1.0/2
trans[(C,B)] = 1.0/2
for i in states:
  inits[i] = 1.0/3

rights = filter(lambda x: x._right, states)
lefts = filter(lambda x: x._left, states)

sequence = 4*np.random.rand(10)-2
sequence = [0.0,1.0,1.0,-1.0,1.0,0.0,0.0,0.0]
for ind in range(len(sequence)):
  fig.add_node(ind, label="%.3f"%sequence[ind], pos="%s,%s!"%(2*ind, 1))


alpha = [{}]
path = {}
for si, s in enumerate(lefts):
  alpha[0][(s, "left")] = inits[s] * s.left().pdf(sequence[0])
  path[(s,"left",0)] = [(s,"left", 0)]
  for ind in range(len(sequence)):
    fig.add_node((s, "left", ind), pos="%s,%s!"%(2*ind, -si), label=s)

for si, s in enumerate(rights, start=len(lefts)):
  alpha[0][(s, "right")] = 0.0
  path[(s,"right",0)] = [(s,"right", 0)]
  for ind in range(len(sequence)):
    fig.add_node((s, "right", ind), pos="%s,%s!"%(2*ind, -si), label=s, style="filled")


print path
for ind in range(1, len(sequence)):
#for ind in range(1, 2):
  a = {}
  newpath = {}
  for s in lefts:
    tmp = {}
    for k,v in alpha[ind-1].iteritems():
      if k[1] == "left" and trans.has_key((k[0], s)):
        #a[(s,"left")] = a.get((s, "left"), 0.0) + v*trans[(k[0], s)]
        tmp[(k[0], k[1], ind-1)] = v*trans.get((k[0], s), 0)
        fig.add_edge((k[0], k[1], ind-1), (s, "left", ind))
    maxa = max(tmp, key=tmp.get)
    node = fig.get_node(maxa)
    node.attr['color'] = "red"
    #a[(s,"left")] = a.get((s, "left"), 0.0) * s.left().pdf(sequence[ind])
    a[(s,"left")] = tmp[maxa] * s.left().pdf(sequence[ind])
    newpath[s] = path[maxa] + [s]
  if ind > 1:
    for s in rights:
      tmp = {}
      for k,v in alpha[ind-1].iteritems():
        if trans.has_key((k[0], s)):
          #a[(s,"right")] = a.get((s, "right"), 0.0) + v*trans[(k[0], s)]
          tmp[(k[0], k[1], ind-1)] = v*trans.get((k[0], s), 0)
          fig.add_edge((k[0], k[1], ind-1), (s, "right", ind))
      #a[(s,"right")] = a.get((s, "right"), 0.0) * s.right().pdf(sequence[ind])
      maxa = max(tmp, key=tmp.get)
      a[(s,"right")] = tmp[maxa] * s.right().pdf(sequence[ind])
      node = fig.get_node(maxa)
      node.attr['color'] = "red"
      newpath[s] = path[maxa] + [s]

  path = newpath
    
  alpha.append(a)

pprint( sequence )
pprint( alpha )
pprint( path )
