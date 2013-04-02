import nltk
import itertools
import numpy as np
from scipy.stats import norm
from pprint import pprint
from nltk.grammar import Nonterminal, WeightedGrammar, parse_grammar, standard_nonterm_parser
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from nltk.parse.pchart import *
from nltk.tree import ProbabilisticTree

import matplotlib.pyplot as plt


class ProbabilisticEmissionRule(AbstractChartRule):
  NUM_EDGES = 3
  def apply_iter(self, chart, grammar, edge):
    if not isinstance(edge, ProbabilisticLeafEdge):
      return
    for state, model in grammar.density().iteritems():
      prob = model(edge.lhs())
      new_edge = ProbabilisticTreeEdge(prob,
                                   (edge.start(), edge.start()),
                                   state, [edge.lhs()], 0)
      if chart.insert(new_edge, ()):
        yield new_edge

class InsideChartSignalParser(nltk.parse.pchart.InsideChartParser):
  def _setprob(self, tree, prod_probs, emission):
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
      prob = emission[lhs](rhs[0])

    for child in tree:
      if isinstance(child, Tree):
        self._setprob(child, prod_probs, emission)
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

    for parse in parses:
      self._setprob(parse, prod_probs, grammar.density())

    parses.sort(reverse=True, key=lambda tree: tree.prob())

    return parses[:n]

class ContinuousWeightedGrammar(WeightedGrammar):
  def __init__(self, emission, density, *args, **kwargs):
    WeightedGrammar.__init__(self, *args, **kwargs)
    self._emission = emission
    self._density = density

  def emission(self):
    return self._emission
  def density(self):
    return self._density
  def sampler(self, n=20):
    return itertools.islice( self.sampling(), n)
  def sampling(self, item=None):
    if not item:
      item = self.start()
    prods = self.productions(lhs = item)
    if len(prods)>0:
      choose = np.random.choice(prods, 1, p=[x.prob() for x in prods])[0]
      for rhs in choose.rhs():
        for s in self.sampling (item=rhs):
          yield s
    else:
      yield (item, self._emission[item](1))

def parse_cpcfg(input, emission, density,  encoding=None):
  start , production = parse_grammar(input, standard_nonterm_parser,probabilistic=True)
  return ContinuousWeightedGrammar(emission,density, start, production)

densityEmission = {
    Nonterminal('top') : lambda x: norm.pdf(x, loc=0, scale=0.3),
    Nonterminal('up') : lambda x: norm.pdf(x, loc=0.2, scale=0.1),
    Nonterminal('down') : lambda x: norm.pdf(x, loc=-0.2, scale=0.1),
}
emission = {
    Nonterminal('top') : lambda x: norm.rvs(loc=0, scale=0.3, size=x),
    Nonterminal('up') : lambda x: norm.rvs(loc=0.2, scale=0.1, size=x),
    Nonterminal('down') : lambda x: norm.rvs(loc=-0.2, scale=0.1, size=x),
}

g = parse_cpcfg("""
  S -> UP TOP DOWN [1.0]
  TOP -> top TOP [0.87] | top [0.13]
  UP -> up UP [0.92] | up [0.08]
  DOWN -> down DOWN [0.92] | down [0.08]
""", emission, densityEmission)

print list(g.sampler(n=40))


toy_pcfg2 = nltk.parse_pcfg("""
    S -> DOWN UP [1.0]
    UP -> up UP [0.8] | up [0.2]
    DOWN -> down DOWN [0.8] | down [0.2]
""")

toy_hmm = nltk.parse_pcfg("""
  S -> UP [1.0]
  UP -> '1' UP [0.92] | TOP [0.08]
  TOP -> '0' TOP [0.87] | DOWN [0.10] | UP [0.03]
  DOWN -> '2' DOWN [0.92] | TOP [0.08]
""")

#text = list("111000222")
#text = [0.1, 0.2, 0.22, 0.05, 0.0, -0.2, -0.11, -0.08]
#parser = InsideChartSignalParser(g)
#parser.trace(2)
#parser.nbest_parse(text)

#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.plot(sample[:,1])
#bx = fig.add_subplot(212)
#bx.plot(np.cumsum(sample[:,1]))
#plt.show()
