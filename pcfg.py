import nltk
import itertools
import numpy as np
from scipy.stats import norm
from pprint import pprint
from nltk.grammar import Nonterminal
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from nltk.parse.pchart import *
from nltk.tree import ProbabilisticTree

import matplotlib.pyplot as plt


MODELS = {
    Nonterminal('top') : lambda x: norm.pdf(x, loc=0, scale=0.3),
    Nonterminal('up') : lambda x: norm.pdf(x, loc=0.2, scale=0.1),
    Nonterminal('down') : lambda x: norm.pdf(x, loc=-0.2, scale=0.1),
}

class ProbabilisticEmissionRule(AbstractChartRule):
  NUM_EDGES = 3
  def apply_iter(self, chart, grammar, edge):
    if not isinstance(edge, ProbabilisticLeafEdge):
      return
    for state, model in MODELS.iteritems():  #FIXME
      prob = model(edge.lhs())
      new_edge = ProbabilisticTreeEdge(prob, 
                                   (edge.start(), edge.start()),
                                   state, [edge.lhs()], 0)
      if chart.insert(new_edge, ()):
        yield new_edge

class ProbabilisticBottomUpPredictRuleDup(AbstractChartRule):
  NUM_EDGES = 1
  def apply_iter(self, chart, grammar, edge):
    if edge.is_incomplete(): return
    for prod in grammar.productions():
      if edge.lhs() == prod.rhs()[0]:
        new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prod.prob())
        if chart.insert(new_edge, ()):
          yield new_edge

class ProbabilisticBottomUpSignalInitRule(AbstractChartRule):
  NUM_EDGES=0
  def apply_iter(self, chart, grammar):
    for index in range(chart.num_leaves()):
      new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
      if chart.insert(new_edge, ()):
        yield new_edge
      #for leaf, model in MODELS.iteritems():  #FIXME
        # given the output chart.leaf(index),
        # figure out the related state and prob
        # prob should be based on the model
        # leaf should be based on the model
        #prob = model(chart.leaf(index))
        #new_edge = ProbabilisticLeafEdge(prob, leaf, index )
        #if chart.insert(new_edge, ()):
        #  yield new_edge

class InsideChartSignalParser(nltk.parse.pchart.InsideChartParser):
  def _setprob(self, tree, prod_probs):
    if tree.prob() is not None : return 
    lhs = Nonterminal(tree.node)
    rhs = []
    for child in tree:
      if isinstance(child, Tree):
        rhs.append(Nonterminal(child.node))
      else:
        rhs.append(child)
    if isinstance(rhs[0], Nonterminal):
      # regular probabilty set from grammar
      prob = prod_probs[lhs, tuple(rhs)]
    else:
      # continuse output
      prob = MODELS[lhs](rhs[0])

    for child in tree:
      if isinstance(child, Tree):
        self._setprob(child, prod_probs)
        prob *= child.prob()

    tree.set_prob(prob)

  def nbest_parse(self, tokens, n=None):
    # now the tokens comes from continuse set
    chart = Chart(list(tokens))
    grammar = self._grammar

    bu_init = ProbabilisticBottomUpInitRule()
    bu = ProbabilisticBottomUpPredictRule()
    fr = SingleEdgeProbabilisticFundamentalRule()
    em = ProbabilisticEmissionRule()

    queue = []

    for edge in bu_init.apply_iter(chart, grammar):
      if self._trace > 1:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
                                edge.prob()))
      queue.append(edge)

    print "finish"
    
    while len(queue) > 0:
      self.sort_queue(queue, chart)

      if self.beam_size:
        self._prune(queue, chart)

      edge = queue.pop()
      if self._trace > 0:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
                                edge.prob()))

      queue.extend(em.apply(chart, grammar, edge))
      queue.extend(bu.apply(chart, grammar, edge))
      queue.extend(fr.apply(chart, grammar, edge))

    parses = chart.parses(grammar.start(), ProbabilisticTree)
    
    prod_probs = {}
    for prod in grammar.productions():
      prod_probs[prod.lhs(), prod.rhs()] = prod.prob()

    pprint (prod_probs)
    for parse in parses:
      self._setprob(parse, prod_probs)

    parses.sort(reverse=True, key=lambda tree: tree.prob())

    return parses[:n]




toy_pcfg = nltk.parse_pcfg("""
  S -> UP TOP DOWN [1.0]
  TOP -> top TOP [0.87] | top [0.13]
  UP -> up UP [0.92] | up [0.08]
  DOWN -> down DOWN [0.92] | down [0.08]
""")

toy_hmm = nltk.parse_pcfg("""
  S -> UP [1.0]
  UP -> '1' UP [0.92] | TOP [0.08]
  TOP -> '0' TOP [0.87] | DOWN [0.10] | UP [0.03]
  DOWN -> '2' DOWN [0.92] | TOP [0.08]
""")

toy_response = {
    0 : (lambda x : np.random.normal(0.0, 0.02, 1)),
    1 : (lambda x : np.random.normal(0.2, 0.01, 1)),
    2 : (lambda x : np.random.normal(-0.2, 0.01, 1)),
}

print toy_pcfg
print toy_hmm
def sampling(grammar, item=None):
  if not item:
    item = grammar.start()
  if isinstance(item, Nonterminal):
    prods = grammar.productions(lhs=item)
    choose = np.random.choice(prods, 1, p=[x.prob() for x in prods])[0]
    for rhs in choose.rhs():
      for s in sampling(grammar, item=rhs):
        yield s
  else:
    item = int(item)
    s = toy_response[item]
    yield (item, s(1))

hmm = itertools.islice( sampling(toy_hmm), 300)
pcfg = itertools.islice( sampling(toy_pcfg), 300)

#text = list("111000222")
text = [0.1, 0.2, 0.22, 0.05, 0.0, -0.2, -0.11, -0.08]
parser = InsideChartSignalParser(toy_pcfg)
parser.trace(2)
for p in parser.nbest_parse(text, n=5):
  print p

#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.plot(sample[:,1])
#bx = fig.add_subplot(212)
#bx.plot(np.cumsum(sample[:,1]))
#plt.show()
