import nltk
import itertools
import numpy as np
from pprint import pprint
from nltk.grammar import Nonterminal
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from nltk.parse.pchart import ProbabilisticBottomUpInitRule, ProbabilisticBottomUpPredictRule, SingleEdgeProbabilisticFundamentalRule
from nltk.tree import ProbabilisticTree

import matplotlib.pyplot as plt

class ProbabilisticLeafEdge(LeafEdge):
  def __init__(self, prob, *args, **kwargs):
    LeafEdge.__init__(self, *args, **kwargs)
    self.prob = prob

  def prob(self):
    return self.prob


MODELS = 1; # given list density
class ProbabilisticBottomUpSignalInitRule(AbstractChartRule):
  NUM_EDGES=0
  def apply_iter(self, chart, grammar):
    for index in range(chart.num_leaves()):
      for model in MODELS:  #FIXME
        # prob should be based on the model
        # leaf should be based on the model
        new_edge = ProbabilisticLeafEdge(prob, leaf, index )
        if chart.insert(new_edge, ()):
          yield new_edge

class InsideChartSignalParser(nltk.parse.pchart.InsideChartParser):
  def nbest_parse(self, tokens, n=None):
    # now the tokens comes from continuse set
    chart = Chart(list(tokens))
    grammar = self._grammar

    bu_init = ProbabilisticBottomUpSignalInitRule()
    bu = ProbabilisticBottomUpPredictRule()
    fr = SingleEdgeProbabilisticFundamentalRule()

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

      queue.extend(bu.apply(chart, grammar, edge))
      queue.extend(fr.apply(chart, grammar, edge))

    parses = chart.parses(grammar.start(), ProbabilisticTree)

    prod_probs = {}
    for prod in grammar.productions():
      prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
    for parse in parses:
      self._setprob(parse, prod_probs)

    parses.sort(reverse=True, key=lambda tree: tree.prob())

    return parses[:n]




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
text = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3]
parser = InsideChartSignalParser(toy_pcfg)
parser.trace(2)
parser.nbest_parse(text)

#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.plot(sample[:,1])
#bx = fig.add_subplot(212)
#bx.plot(np.cumsum(sample[:,1]))
#plt.show()
