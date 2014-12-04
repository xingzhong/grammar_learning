import argparse
import csv
from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal
import pandas as pd
import numpy as np

def grammarWithNA():
	S = Nonterminal('S')
	S1 = Nonterminal('S1')
	S2 = Nonterminal('S2')
	HS = Nonterminal('NT-HS')
	IHS = Nonterminal('NT-IHS')
	NA = Nonterminal('NA')

	hs = Terminal('HS', None, None)
	ihs = Terminal('IHS', None, None)
	btop = Terminal('BTOP', None, None)
	bbot = Terminal('BBOT', None, None)
	ber = Terminal('BER', None, None)
	bet = Terminal('BET', None, None)
	bur = Terminal('BUR', None, None)
	but = Terminal('BUT', None, None)
	fw = Terminal('FW', None, None)
	fc = Terminal('FC', None, None)
	rw = Terminal('RW', None, None)
	rc = Terminal('RC', None, None)
	na = Terminal('N/A', None, None)

	prods = [ ]
	prods.append(Production(S, [S1, S], prob=.5))
	prods.append(Production(S, [S1, None], prob=.5))
	prods.append(Production(S1, [S2, NA], prob=.5))
	prods.append(Production(S1, [NA, S2], prob=.5))
	prods.append(Production(S2, [hs, None], prob=1/12.0))
	prods.append(Production(S2, [ihs, None], prob=1/12.0))
	prods.append(Production(S2, [btop, None], prob=1/12.0))
	prods.append(Production(S2, [bbot, None], prob=1/12.0))
	prods.append(Production(S2, [ber, None], prob=1/12.0))
	prods.append(Production(S2, [bet, None], prob=1/12.0))
	prods.append(Production(S2, [bur, None], prob=1/12.0))
	prods.append(Production(S2, [but, None], prob=1/12.0))
	prods.append(Production(S2, [fw, None], prob=1/12.0))
	prods.append(Production(S2, [fc, None], prob=1/12.0))
	prods.append(Production(S2, [rw, None], prob=1/12.0))
	prods.append(Production(S2, [rc, None], prob=1/12.0))
	prods.append(Production(NA, [na, NA], prob=.6))
	prods.append(Production(NA, [na, None], prob=.4))

	return Grammar(S, prods)

def sampleGrammar():
	S = Nonterminal('S')
	S1 = Nonterminal('S1')
	C = Nonterminal('C')
	C1 = Nonterminal('C1')
	R = Nonterminal('R')
	R1 = Nonterminal('R1')

	hs = Terminal('HS', None, None)
	ihs = Terminal('IHS', None, None)
	btop = Terminal('BTOP', None, None)
	bbot = Terminal('BBOT', None, None)
	ttop = Terminal('TTOP', None, None)
	tbot = Terminal('TBOT', None, None)

	prods = [ ]
	prods.append(Production(S, [S1, S], prob=.5))
	prods.append(Production(S, [S1, None], prob=.5))
	prods.append(Production(S1, [hs, None], prob=1.0/6))
	prods.append(Production(S1, [ihs, None], prob=1.0/6))
	prods.append(Production(S1, [btop, None], prob=1.0/6))
	prods.append(Production(S1, [bbot, None], prob=1.0/6))
	prods.append(Production(S1, [ttop, None], prob=1.0/6))
	prods.append(Production(S1, [tbot, None], prob=1.0/6))
	

	return Grammar(S, prods)

def grammar():
	S = Nonterminal('S')
	S1 = Nonterminal('S1')
	C = Nonterminal('C')
	c = Nonterminal('c')

	hs = Terminal('hs', None, None)
	ihs = Terminal('ihs', None, None)
	btop = Terminal('btop', None, None)
	bbot = Terminal('bbot', None, None)
	ttop = Terminal('ttop', None, None)
	tbot = Terminal('tbot', None, None)

	prods = [ ]
	prods.append(Production(S, [S1, S], prob=.5))
	prods.append(Production(S, [S1, None], prob=.5))
	prods.append(Production(S1, [C, hs], prob=.25))
	prods.append(Production(S1, [C, ihs], prob=.25))
	prods.append(Production(S1, [C, btop], prob=.25))
	prods.append(Production(S1, [C, bbot], prob=.25))
	prods.append(Production(C, [c, C], prob=.4))
	prods.append(Production(C, [c, None], prob=.6))
	prods.append(Production(c, [ttop, None], prob=.5))
	prods.append(Production(c, [tbot, None], prob=.5))



	return Grammar(S, prods)

def readDf(f):
	return pd.read_csv(f, index_col=0, parse_dates=0)
	 
def buildTable(ptables):
	length = 0
	d = {}
	beta = 2.0
	#import ipdb; ipdb.set_trace()
	for idx, (row_index, row) in enumerate(ptables.iterrows()):
		for t, p in row.iteritems():
			s, e = idx, idx + 4
			if int(e) > length: length = int(e)
			d[(int(s), int(e), Terminal(t, None, None))] = float(p)
			d[(int(s), int(s), Terminal('N/A', None, None))] = -beta
			d[(int(s+1), int(s+1), Terminal('N/A', None, None))] = -beta
			d[(int(s+2), int(s+2), Terminal('N/A', None, None))] = -beta
			d[(int(s+3), int(s+3), Terminal('N/A', None, None))] = -beta
			d[(int(s+4), int(s+4), Terminal('N/A', None, None))] = -beta
	return d, length, ptables.index

def t2d(t):
	# table to dict
	d = {}
	length = 0

	with open(t, 'rb') as f:
		reader = csv.reader(f)
		for ter, s, e, l in reader:
			if int(e) > length: length = int(e)
			d[(int(s), int(e), Terminal(ter, None, None))] = float(l)	
	return d, length


def parse(ptables):
	pdict, length, idx = buildTable(ptables)
	#g = grammar()
	#g = sampleGrammar()
	g = grammarWithNA()
	model = PCFG(g)
	#import ipdb; ipdb.set_trace()
	lik, tree = model.ptable(pdict, length)
	#for k, v in model.viterbi_.iteritems():
	#	if not np.isinf(v):
	#		print k, v
	decodes = []
	d = []
	for x in tree.BFS():
		print x
		if isinstance(x[0], Terminal):
			d.append(x)	
	return d
			

def main():
	parser = argparse.ArgumentParser(description="parsing from the ptable")
	parser.add_argument('-t', '--table', help='ptable likelihood', required=True)
	args = vars(parser.parse_args())
	ptable = readDf('test.csv')
	parse(ptable)
	

if __name__ == '__main__':
	main()

	

