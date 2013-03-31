import nltk
import numpy as np
from pprint import pprint
from nltk.grammar import Nonterminal

import matplotlib.pyplot as plt

toy_pcfg = nltk.parse_pcfg("""
  S -> UP TOP DOWN [1.0]
  TOP -> '0' TOP [0.87] | '0' [0.13]
  UP -> '1' UP [0.92] | '1' [0.08]
  DOWN -> '2' DOWN [0.92] | '2' [0.08]
""")

toy_hmm = nltk.parse_pcfg("""
  S -> UP [1.0]
  UP -> '1' UP [0.92] | TOP [0.08]
  TOP -> '0' TOP [0.87] | DOWN [0.10] | UP [0.03]
  DOWN -> '2' DOWN [0.92] | '2' [0.08]
""")

toy_response = {
    0 : (lambda x : np.random.normal(0.0, 0.02, 1)),
    1 : (lambda x : np.random.normal(0.2, 0.01, 1)),
    2 : (lambda x : np.random.normal(-0.2, 0.01, 1)),
}

print toy_pcfg
print toy_hmm
sample = []
def sampling(grammar, item=None):
  if not item:
    item = grammar.start()
  if isinstance(item, Nonterminal):
    prods = grammar.productions(lhs=item)
    choose = np.random.choice(prods, 1, p=[x.prob() for x in prods])[0]
    for rhs in choose.rhs():
      sampling(grammar, item=rhs)
  else:
    item = int(item)
    s = toy_response[item]
    sample.append ((item, s(1)))

sampling(toy_hmm)
print sample
sample = np.array(sample)
print sample[:,1]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(sample[:,1])
bx = fig.add_subplot(212)
bx.plot(np.cumsum(sample[:,1]))
plt.show()
