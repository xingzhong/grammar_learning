from BiCluster import *
import numpy as np

def learn(sample):
	sample2 = sample.copy().tolist()
	g = None
	bcs = []
	totalBits = sum(map(len, sample2))
	for i in range(30):
		print "alpha:%s\n"%(sum(map(len, sample2))/float(totalBits))
		table, symbols = s2s(sample2)
		ecm, cols, _ = buildEcm(sample2)
		bc = bestBC(table, symbols, ecm, cols)
		if np.isneginf(bc.logGain()): 
			print "no more !"
			break
		bcs.append(bc)
		new = 'NT_%s'%i
		sample2 = bc.reduction(sample2, new)
		prods = bc.toRules(new)
		g = addProd(g, prods, new)
		print bc
		table, symbols = s2s(sample2)
		ecm, cols, _ = buildEcm(sample2)
		for _bc in bcs:
			bc_new = BiCluster().update(_bc, table, ecm, col=Nonterminal(new))
			#print "bcG: %s"%bc_new.logGain()
			if bc_new and bc_new.logGain() > 10.0:
				print "Adding col %s to %s"%(new, bc_new._nt)
				print bc_new
				g = addProd(g, bc_new.toRules(), bc_new._nt)
				sample2 = bc_new.reduction(sample2)
				bcs.append(bc_new)
				continue
			bc_new = BiCluster().update(_bc, table, ecm, row=Nonterminal(new))
			#print "bcG: %s"%bc_new.logGain()
			if bc_new and bc_new.logGain() > 10.0:
				print "Adding col %s to %s"%(new, bc_new._nt)
				print bc_new
				g = addProd(g, bc_new.toRules(), bc_new._nt)
				sample2 = bc_new.reduction(sample2)
				bcs.append(bc_new)
				continue
	g = postProcess(g, sample2)
	print "finish"
	return g, sample2

def main():
    import string
    sample = np.random.choice(list(string.lowercase[:4]), (30,10))
    sample = np.asarray(sample, dtype= np.dtype("object") )
    g, sample2 = learn(sample)
    save(sample, g)

def save(sample, g):
	import pickle
	pickle.dump((sample, g), open("data.p", "wb"))
	print "saving"

def load():
	import pickle
	print "loading"
	return pickle.load(open("data.p", "rb"))

def parsing(sample, g):
	from nltk.parse.viterbi import ViterbiParser
	from nltk.draw.tree import draw_trees
	parser = ViterbiParser(g)
	for s in sample:
		print " ".join(s)
		t = parser.parse(s)
		print t
		if t:
			draw_trees(t)

if __name__ == '__main__':
    main()
    #sample, g = load()
    #print g
    #parsing(sample, g)
    