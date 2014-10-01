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

class CProbabilisticLeafEdge(ProbabilisticLeafEdge):
  def logprob(self): return 0;

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
        prob += child.prob()

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

if __name__ == "__main__":
  #TODO: use current HMM example to verify the results
  densityEmission = {
      Nonterminal('s1') : lambda x: norm.logpdf(x, loc=0, scale=1),
      Nonterminal('s2') : lambda x: norm.logpdf(x, loc=20, scale=1),
      Nonterminal('s3') : lambda x: norm.logpdf(x, loc=-20, scale=1),
  }
  emission = {
      Nonterminal('s1') : lambda x: norm.rvs(loc=0, scale=1, size=x),
      Nonterminal('s2') : lambda x: norm.rvs(loc=20, scale=1, size=x),
      Nonterminal('s3') : lambda x: norm.rvs(loc=-20, scale=1, size=x),
  }

  g = parse_cpcfg("""
    S -> S1 [0.6] | S2 [0.3]| S3 [0.1]
    S1 -> s1 S1 [0.7] | S1 S2 [0.2] | S1 S3[0.0999] | s1 [0.0001]
    S2 -> S2 S1 [0.7] | s2 S2 [0.2] | S2 S3[0.0999] | s2 [0.0001]
    S3 -> S3 S1 [0.7] | S3 S2 [0.2] | s3 S3[0.0999] | s3 [0.0001]
    """, emission, densityEmission)
  # seems like each production has to have one output.

  from sklearn import hmm
  startprob = np.array([0.6, 0.3, 0.1])
  transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
  means = np.array([[0], [20], [-20]])
  covars = np.tile(np.identity(1), (3,1,1))
  model = hmm.GaussianHMM(3, 'full', startprob, transmat)
  model.means_ = means
  model.covars_ = covars
  X, Z = model.sample(6)

  parser = InsideChartSignalParser(g)
  parser.trace(3)
  X = list(X.flat)
  #parses = parser.nbest_parse([0, 0], n=3)
  # cannot parse it well TODO
  #for p in parses:
  #  ContextFreeGrammarprint p
  #  print p.pprint(parens=u'[]')

  #print list(g.sampler(n=10))
