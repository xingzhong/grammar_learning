from __future__ import print_function, unicode_literals
from nltk.grammar import WeightedGrammar, Nonterminal, WeightedProduction
from nltk.parse.pchart import InsideChartParser
from nltk.compat import string_types, python_2_unicode_compatible
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from nltk.tree import Tree, ProbabilisticTree

from pcfg import Dist
import numpy as np
import re

class ProbabilisticLeafEdge(LeafEdge):
    def prob(self): 
        #print ("%s, %s"%self._comparison_key)
        return 1.0

class ProbabilisticTreeEdge(TreeEdge):
    def __init__(self, prob, *args, **kwargs):
        TreeEdge.__init__(self, *args, **kwargs)
        self._prob = prob
        # two edges with different probabilities are not equal.
        self._comparison_key = (self._comparison_key, prob)

    def prob(self): return self._prob

    @staticmethod
    def from_production(production, index, p):
        return ProbabilisticTreeEdge(p, (index, index), production.lhs(),
                                     production.rhs(), 0)

# Rules using probabilistic edges
class ProbabilisticBottomUpInitRule(AbstractChartRule):
    NUM_EDGES=0
    def apply_iter(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge

class ProbabilisticBottomUpPredictRule(AbstractChartRule):
    NUM_EDGES=1
    def apply_iter(self, chart, grammar, edge):
        if edge.is_incomplete(): return
        for prod in grammar.productions():
            if isinstance(prod.rhs()[0], Dist) and not isinstance(edge.lhs(), Nonterminal) :
                logProb = prod.rhs()[0].logPdf(edge.lhs())[0][0]
                prob = np.exp(logProb)
                prob = prob * prod.prob()
                new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prob)
                if chart.insert(new_edge, ()):
                    yield new_edge
            if edge.lhs() == prod.rhs()[0]:
                new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prod.prob())
                if chart.insert(new_edge, ()):
                    yield new_edge

class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=2
    def apply_iter(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        
        if left_edge.end() != right_edge.start() :
            return 
        if not (left_edge.is_incomplete() and right_edge.is_complete()):
            return
        if not isinstance( left_edge.nextsym(), Dist ) :
            if not (left_edge.nextsym() == right_edge.lhs()) :
                return 
        
        
        # Construct the new edge.
        #import pdb; pdb.set_trace()
        if isinstance( left_edge.nextsym(), Dist ) and not isinstance(right_edge.lhs(), Nonterminal):
            p = left_edge.prob() * right_edge.prob() 
            p = p * np.exp( left_edge.nextsym().logPdf(right_edge.lhs())[0][0] )
        else :
            p = left_edge.prob() * right_edge.prob()
        new_edge = ProbabilisticTreeEdge(p,
                            span=(left_edge.start(), right_edge.end()),
                            lhs=left_edge.lhs(), rhs=left_edge.rhs(),
                            dot=left_edge.dot()+1)

        # Add it to the chart, with appropriate child pointers.
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1+(right_edge,)):
                changed_chart = True

        # If we changed the chart, then generate the edge.
        if changed_chart: 
            print (left_edge)
            print (right_edge)
            print (new_edge)
            import pdb; pdb.set_trace()
            yield new_edge

@python_2_unicode_compatible
class SingleEdgeProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=1

    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply_iter(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        #import pdb; pdb.set_trace()
        if edge1.is_incomplete():
            # edge1 = left_edge; edge2 = right_edge
            for edge2 in chart.select(start=edge1.end(), is_complete=True,
                                     lhs=edge1.nextsym()):
            #for edge2 in chart.select(start=edge1.end(), is_complete=True):
                #import pdb; pdb.set_trace()
                for new_edge in fr.apply_iter(chart, grammar, edge1, edge2):
                    yield new_edge
        else:
            # edge2 = left_edge; edge1 = right_edge
            #import pdb; pdb.set_trace()
            for edge2 in chart.select(end=edge1.start(), is_complete=False):
            #for edge2 in chart.select(end=edge1.start(), is_complete=False,
            #                          nextsym=edge1.lhs()):
                
                for new_edge in fr.apply_iter(chart, grammar, edge2, edge1):
                    yield new_edge

    def __str__(self):
        return 'Fundamental Rule'

class Parser(InsideChartParser):
    def nbest_parse(self, tokens, n=None):
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        grammar = self._grammar

        # Chart parser rules.
        bu_init = ProbabilisticBottomUpInitRule()
        bu = ProbabilisticBottomUpPredictRule()
        fr = SingleEdgeProbabilisticFundamentalRule()

        # Our queue!
        queue = []

        # Initialize the chart.
        for edge in bu_init.apply_iter(chart, grammar):
            if self._trace > 1:
                print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
                                        edge.prob()))
            queue.append(edge)
        
        print ("Initialize the chart.")

        while len(queue) > 0:
            #print (len(queue))
            # Re-sort the queue.

            self.sort_queue(queue, chart)

            # Prune the queue to the correct size if a beam was defined
            if self.beam_size:
                self._prune(queue, chart)

            # Get the best edge.
            edge = queue.pop()
            if self._trace > 0:
                print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
                                        edge.prob()))

            # Apply BU & FR to it.
            queue.extend(bu.apply(chart, grammar, edge))
            print ("[after bu] %s"%len(queue))
            #import pdb; pdb.set_trace()
            queue.extend(fr.apply(chart, grammar, edge))
            print ("[after fr] %s"%len(queue))
            print ("chart, %s"%(len(chart.edges())))
            #import pdb; pdb.set_trace()

        # Get a list of complete parses.
        parses = chart.parses(grammar.start(), ProbabilisticTree)

        # Assign probabilities to the trees.
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        for parse in parses:
            self._setprob(parse, prod_probs)

        # Sort by probability
        parses.sort(reverse=True, key=lambda tree: tree.prob())

        return parses[:n]

class Grammar(WeightedGrammar):
    def check_coverage(self, tokens):
        return True


if __name__ == '__main__':
    S = Nonterminal("S")
    NT1 = Nonterminal("NT1")
    A = Dist(np.array([2.0]), np.array([1.0]))
    p1 = WeightedProduction(S, [NT1], prob=0.3)
    p2 = WeightedProduction(S, [NT1, S], prob=0.7)
    p3 = WeightedProduction(NT1, [A], prob=1.0)
    #p3 = WeightedProduction(NT1, ['0'], prob=1.0)
    g = Grammar(S, [p1, p2, p3])
    parser = Parser(g) 
    print (g)
    
    parser.trace(3)
    #obs = ['0']*4
    obs = [0] * 4
    parses = parser.nbest_parse(obs)

    for p in parses:
        print (p)