from BiCluster import *
import numpy as np

def learn(sample):
	sample2 = sample.copy().tolist()
	g = None
	bcs = []
	totalBits = sum(map(len, sample2))
	for i in range(100):
		print "alpha:%s\n"%(sum(map(len, sample2))/float(totalBits))
		table, symbols = s2s(sample2)
		ecm, cols, _ = buildEcm(sample2)
		bc = bestBC(table, symbols, ecm, cols)
		if not bc: 
			print "no more rules!"
			break
		bcs.append(bc)
		new = 'NT_%s'%i
		sample2 = bc.reduction(sample2, new)
		prods = bc.toRules(new)
		g = addProd(g, prods, new)
		print "new"
		print bc
		table, symbols = s2s(sample2)
		ecm, cols, _ = buildEcm(sample2)

		for ind, _bc in enumerate(bcs):
			bc_new_c = BiCluster().update(_bc, table, ecm, col=Nonterminal(new))
			bc_new_r = BiCluster().update(_bc, table, ecm, row=Nonterminal(new))
			#print "bcG: %s"%bc_new.logGain()
			best = None
			if bc_new_c :
				bc_new = bc_new_c 
				best = bc_new_c.logGain()
			if bc_new_r and bc_new_r.logGain() > best:
				bc_new = bc_new_r
				best = bc_new_r.logGain()
			if best - bc.logGain() > 2.0:
				print "Attach"
				print bc_new
				g = addProd(g, bc_new.toRules(), bc_new._nt)
				sample2 = bc_new.reduction(sample2)
				bcs[ind] = bc_new
				
	g = postProcess(g, sample2)
	print "finish"
	return g, sample2

def main():
    import string
    #sample = np.random.choice(list(string.lowercase[:6]), (50,20))
    sample = np.random.choice(["A", "T", "C", "G"], (100,10), p=[0.1, 0.3, 0.5, 0.1])
    sample = np.asarray(sample, dtype= np.dtype("object") )
    g, sample2 = learn(sample)
    save(sample, sample2, g)

def save(sample, sample2, g):
	import pickle
	pickle.dump((sample, sample2, g), open("data.p", "wb"))
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
		if t:
			print t.logprob()
			#draw_trees(t)

if __name__ == '__main__':
	main()
	sample, sample2, g = load()
	#print sample
	print sample2
	print g
	#parsing(sample, g)
    