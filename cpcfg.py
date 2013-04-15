import nltk
from nltk.parse.cpchart import CInsideChartParser
from nltk.grammar import parse_cpcfg, Nonterminal

from scipy.stats import norm
from matplotlib.finance import quotes_historical_yahoo as yahoo

if __name__ == "__main__":
  import datetime
  import numpy as np
  date1 = datetime.date(2010, 10, 13)
  date2 = datetime.date(2010, 11, 26)
  quotes = yahoo("uup", date1, date2)
  if (len(quotes) == 0):
    raise SystemExit
  test = np.diff([100 * np.log(q[2]) for q in quotes])
  test = np.round(test, decimals=3).tolist()
  print test
  #test = [-0.5,-0.5, 0,0, 0.5, 0.5]
  densityEmission = {
      Nonterminal('e1') : lambda x : norm.logpdf(x, loc=0, scale=1),
      Nonterminal('e2') : lambda x : norm.logpdf(x, loc=0.5, scale=1),
      Nonterminal('e3') : lambda x : norm.logpdf(x, loc=-0.5, scale=1),
      }
  emission = {
      Nonterminal('e1') : lambda x : norm.rvs(x, loc=0, scale=1),
      Nonterminal('e2') : lambda x : norm.rvs(x, loc=0.5, scale=1),
      Nonterminal('e3') : lambda x : norm.rvs(x, loc=-0.5, scale=1),
      }
  grammar = parse_cpcfg("""
    S -> UPTRI [0.5] | BOTRI [0.5]
    UPTRI -> UP HEAD DOWN [1.0]
    BOTRI -> DOWN HEAD UP [1.0]
    HEAD -> e1 HEAD [0.9] | e1 [0.1]
    UP -> e2 UP [0.9] | e2 [0.1]
    DOWN -> e3 DOWN [0.9] | e3 [0.1]
    """, emission, densityEmission)
  grammar_M = parse_cpcfg("""
    S -> UPTRI [0.5] | BOTRI [0.5]
    UPTRI -> e2 UPTRI e3 [0.9] | HEAD [0.1]
    BOTRI -> e3 BOTRI e2 [0.9] | HEAD [0.1]
    HEAD -> e1 HEAD [0.8] | e1 [0.2]
    """, emission, densityEmission)
  #parser = CInsideChartParser(grammar, beam_size = 2000)
  #parser = CInsideChartParser(grammar_M, beam_size = 2000)
  parser = CInsideChartParser(grammar_M)
  parser.trace(1)
  parsers = parser.nbest_parse(test, n=3)
  for p in parsers:
    print (p.pprint(parens=u'[]'))
    print (p.logprob())
  print (test)
