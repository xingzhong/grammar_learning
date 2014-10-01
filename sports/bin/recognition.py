import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.ndimage.filters import convolve1d
from sklearn import mixture
from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal
from sklearn.mixture.gmm import log_multivariate_normal_density
from itertools import combinations, permutations

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return convolve1d(interval, window, axis=0)

def cart2polar(matrix, center):
	# matrix should have 2 column 
	assert matrix.shape[1] == 2
	m = matrix - center
	r = np.sqrt(np.sum(m**2,axis=1))
	theta = np.arctan2(m[:,0], m[:,1])
	return np.column_stack((r, theta))

def pickVector(polar, team):

	m,_,_ = polar.shape
	polar = np.swapaxes(polar, 0, 1)
	attack = polar[team==True]
	vectors = []
	for at1, at2 in permutations(range(attack.shape[0]), 2):
		atk1, atk2 = attack[at1], attack[at2]
		picker0 = atk2 - atk1
		vectors.append(picker0)
	vectors = np.dstack(vectors)
	#import ipdb; ipdb.set_trace()
	
	return vectors.reshape(-1,4)

def defVector(polar, team):
	polar = np.swapaxes(polar, 0, 1)
	attack = polar[team==True]
	defence = polar[team==False]
	vectors = []
	for att in attack:
		vectors.append(defence-att)
	
	vectors = np.vstack(vectors)
	vectors = np.swapaxes(vectors, 0, 1)
	return vectors.reshape(-1,8)
	#return np.sum(scores, axis=0)

def PR_tactic4():
	S = Nonterminal('S')
	PR = Nonterminal('PR')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	pick = Nonterminal('pick')
	NULL = Nonterminal('NULL')
	Block = Nonterminal('Block')
	Defence = Nonterminal('Defence')

	gBlock = mixture.GMM(n_components=1)
	gBlock.means_ = np.array([[120]])
	gBlock.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), gBlock))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), gBlock, -10))

	gPick = mixture.GMM(n_components=2)
	gPick.means_ = np.array([[100, .3], [100, -.3]])
	gPick.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	gPick.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], gPick))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], gPick))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], gPick, -300))

	gDef = mixture.GMM(n_components=1)
	gDef.means_ = np.array([[-60, 0]])
	gDef.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), gDef))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), gDef, -450))

	prods = [ ]
	prods.append(Production(S, [NULL, PR], prob=1.))
	prods.append(Production(PR, [PICK, Defence], prob=1.))
	prods.append(Production(PICK, [Pick, Block], prob=1.))
	prods.append(Production(Defence, [Defence, defBad], prob=.8))
	prods.append(Production(Defence, [defBad, defBad], prob=.2))
	prods.append(Production(Pick, [pick, Pick], prob=.8))
	prods.append(Production(Pick, [pick, pick], prob=.2))
	prods.append(Production(pick, [Pick0, Pick0], prob=.5))
	prods.append(Production(pick, [Pick1, Pick1], prob=.5))
	prods.append(Production(Block, [block, Block], prob=.8))
	prods.append(Production(Block, [block, block], prob=.2))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))

	return Grammar(S, prods)

def PR_tactic1():
	S = Nonterminal('S')
	PR = Nonterminal('PR')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	Pick0 = Nonterminal('Pick0')
	Pick1 = Nonterminal('Pick1')
	pick = Nonterminal('pick')
	pick0 = Nonterminal('pick0')
	pick1 = Nonterminal('pick1')
	NULL = Nonterminal('NULL')
	Block = Nonterminal('Block')
	Defence = Nonterminal('Defence')

	gBlock = mixture.GMM(n_components=1)
	gBlock.means_ = np.array([[105]])
	gBlock.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), gBlock))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), gBlock, -100))

	gPick = mixture.GMM(n_components=2)
	gPick.means_ = np.array([[100, .3], [100, -.3]])
	gPick.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	gPick.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], gPick))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], gPick))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], gPick, -200))

	gDef = mixture.GMM(n_components=1)
	gDef.means_ = np.array([[-60, 0]])
	gDef.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), gDef))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), gDef, -280))

	prods = [ ]
	prods.append(Production(S, [NULL, PR], prob=1.))
	prods.append(Production(PR, [PICK, Defence], prob=1.))
	prods.append(Production(PICK, [Pick0, Block], prob=.5))
	prods.append(Production(PICK, [Pick1, Block], prob=.5))
	prods.append(Production(Defence, [Defence, defBad], prob=.8))
	prods.append(Production(Defence, [defBad, defBad], prob=.2))
	prods.append(Production(Pick0, [pick0, Pick], prob=.8))
	prods.append(Production(Pick0, [pick0, pick0], prob=.2))
	prods.append(Production(Pick1, [pick1, Pick], prob=.8))
	prods.append(Production(Pick1, [pick1, pick1], prob=.2))
	prods.append(Production(pick0, [Pick0, Pick0], prob=1.))
	prods.append(Production(pick1, [Pick1, Pick1], prob=1.))
	prods.append(Production(Block, [block, Block], prob=.8))
	prods.append(Production(Block, [block, block], prob=.2))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))

	return Grammar(S, prods)

def PR_tactic3():
	S = Nonterminal('S')
	PR = Nonterminal('PR')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	Pick0 = Nonterminal('Pick0')
	Pick1 = Nonterminal('Pick1')
	pick = Nonterminal('pick')
	pick0 = Nonterminal('pick0')
	pick1 = Nonterminal('pick1')
	NULL = Nonterminal('NULL')
	Block = Nonterminal('Block')
	Defence = Nonterminal('Defence')

	gBlock = mixture.GMM(n_components=1)
	gBlock.means_ = np.array([[105]])
	gBlock.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), gBlock))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), gBlock, -100))

	gPick = mixture.GMM(n_components=2)
	gPick.means_ = np.array([[100, .3], [100, -.3]])
	gPick.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	gPick.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], gPick))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], gPick))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], gPick, -200))

	gDef = mixture.GMM(n_components=1)
	gDef.means_ = np.array([[-60, 0]])
	gDef.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), gDef))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), gDef, -1080))

	prods = [ ]
	prods.append(Production(S, [NULL, PR], prob=1.))
	prods.append(Production(PR, [PICK, Defence], prob=1.))
	prods.append(Production(PICK, [Pick0, Block], prob=.5))
	prods.append(Production(PICK, [Pick1, Block], prob=.5))
	prods.append(Production(Defence, [Defence, defBad], prob=.8))
	prods.append(Production(Defence, [defBad, defBad], prob=.2))
	prods.append(Production(Pick0, [pick0, Pick], prob=.8))
	prods.append(Production(Pick0, [pick0, pick0], prob=.2))
	prods.append(Production(Pick1, [pick1, Pick], prob=.8))
	prods.append(Production(Pick1, [pick1, pick1], prob=.2))
	prods.append(Production(pick0, [Pick0, Pick0], prob=1.))
	prods.append(Production(pick1, [Pick1, Pick1], prob=1.))
	prods.append(Production(Block, [block, Block], prob=.8))
	prods.append(Production(Block, [block, block], prob=.2))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))

	return Grammar(S, prods)

def PR_tactic7():
	S = Nonterminal('S')
	PR = Nonterminal('PR')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	Pick0 = Nonterminal('Pick0')
	Pick1 = Nonterminal('Pick1')
	pick = Nonterminal('pick')
	pick0 = Nonterminal('pick0')
	pick1 = Nonterminal('pick1')
	NULL = Nonterminal('NULL')
	Block = Nonterminal('Block')
	Defence = Nonterminal('Defence')

	gBlock = mixture.GMM(n_components=1)
	gBlock.means_ = np.array([[95]])
	gBlock.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), gBlock))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), gBlock, -50))

	gPick = mixture.GMM(n_components=2)
	gPick.means_ = np.array([[100, .3], [100, -.3]])
	gPick.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	gPick.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], gPick))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], gPick))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], gPick, -200))

	gDef = mixture.GMM(n_components=1)
	gDef.means_ = np.array([[-60, 0]])
	gDef.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), gDef))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), gDef, -500))

	prods = [ ]
	prods.append(Production(S, [NULL, PR], prob=1.))
	prods.append(Production(PR, [PICK, Defence], prob=1.))
	prods.append(Production(PICK, [Pick0, Block], prob=.5))
	prods.append(Production(PICK, [Pick1, Block], prob=.5))
	prods.append(Production(Defence, [Defence, defBad], prob=.8))
	prods.append(Production(Defence, [defBad, defBad], prob=.2))
	prods.append(Production(Pick0, [pick0, Pick], prob=.8))
	prods.append(Production(Pick0, [pick0, pick0], prob=.2))
	prods.append(Production(Pick1, [pick1, Pick], prob=.8))
	prods.append(Production(Pick1, [pick1, pick1], prob=.2))
	prods.append(Production(pick0, [Pick0, Pick0], prob=1.))
	prods.append(Production(pick1, [Pick1, Pick1], prob=1.))
	prods.append(Production(Block, [block, Block], prob=.8))
	prods.append(Production(Block, [block, block], prob=.2))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))

	return Grammar(S, prods)


def positive(x, mask, g):
	#import ipdb; ipdb.set_trace()
	return np.atleast_2d(g.score(x[:, mask])).T
def negative(x, mask, g, threshold):
	pos = np.array((positive(x, [0,2], g), positive(x, [1,3], g)))
	#import ipdb; ipdb.set_trace()
	return threshold-np.max(pos, axis=0)
	
def poly_area2D(poly):
	total = 0.0
	N = len(poly)
	for i in range(N):
		v1 = poly[i]
		v2 = poly[(i+1) % N]
		total += v1[0]*v2[1] - v1[1]*v2[0]
	return np.log(abs(total/2))

def poly_dist(poly):
	d = 0.0
	for i, j in combinations(range(4), 2):
		d += np.linalg.norm(poly[i] - poly[j])
	#import ipdb; ipdb.set_trace()
	return d/6.0

def blockBad(x, mask, g, threshold):
	return threshold-blockGood(x, mask, g)

def blockGood(x, mask, g):
	#areas = map(poly_area2D, x[:,mask].reshape(-1, 4, 2))
	areas = map(poly_dist, x[:,mask].reshape(-1, 4, 2))
	areas = np.atleast_2d(areas).T
	#import ipdb; ipdb.set_trace()
	scores = np.atleast_2d(g.score(areas)).T
	
	return scores
	#import ipdb; ipdb.set_trace()
	#return

def blockGrammar():
	S = Nonterminal('BLOCK')
	Block = Nonterminal('Block')
	g = mixture.GMM(n_components=1)
	g.means_ = np.array([[4]])
	g.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), g))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), g))

	prods = [ ]
	prods.append(Production(S, [Block, S], prob=.8))
	prods.append(Production(S, [Block, Block], prob=.2))
	prods.append(Production(Block, [block, block], prob=.5))
	prods.append(Production(Block, [nonBlock, nonBlock], prob=.5))

	return Grammar(S, prods)

def defGrammar():
	S = Nonterminal('DEF')
	Defence = Nonterminal('Defence')
	#PickN = Terminal('N/A', 
	#	np.array([[0, 0, 0, 0]]), np.array([[1e5, 1e5, 1, 1]]))
	#Pick0 = Terminal('picker0', 
	#	np.array([[100, 0, 0, 0]]), np.array([[10, 1e5, 1e-3, 1]]))

	g = mixture.GMM(n_components=1)
	g.means_ = np.array([[-70, 0]])
	g.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), g))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), g))

	prods = [ ]
	prods.append(Production(S, [Defence, S], prob=.8))
	prods.append(Production(S, [Defence, Defence], prob=.2))
	prods.append(Production(Defence, [defGood, defGood], prob=.5))
	prods.append(Production(Defence, [defBad, defBad], prob=.5))

	return Grammar(S, prods)

def defenceGood(x, mask, g):
	scores = g.score(x[:, mask].reshape(-1, 2)).reshape(-1, 2, 2)
	scoresAtt = np.max(scores, axis=2)
	scoresDef = np.sum(scoresAtt, axis=1)
	#
	#plt.plot(scoresAtt); plt.show()
	#import ipdb; ipdb.set_trace()
	return np.atleast_2d(scoresDef).T
	
def defenceBad(x, mask, g, threshold):
	return threshold-defenceGood(x, mask, g)


def pickGrammar():

	S = Nonterminal('S')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	NULL = Nonterminal('NULL')
	#PickN = Terminal('N/A', 
	#	np.array([[0, 0, 0, 0]]), np.array([[1e5, 1e5, 1, 1]]))
	#Pick0 = Terminal('picker0', 
	#	np.array([[100, 0, 0, 0]]), np.array([[10, 1e5, 1e-3, 1]]))

	g = mixture.GMM(n_components=2)
	g.means_ = np.array([[100, .3], [100, -.3]])
	g.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	g.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], g))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], g))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], g))

	

	prods = [ ]
	prods.append(Production(S, [NULL, PICK], prob=1.))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))
	prods.append(Production(PICK, [Pick, PICK], prob=.8))
	prods.append(Production(PICK, [Pick, Pick], prob=.2))
	prods.append(Production(Pick, [Pick0, Pick0], prob=.5))
	prods.append(Production(Pick, [Pick1, Pick1], prob=.5))

	return Grammar(S, prods)

def heat4():
	from template import Pts4 as HOOP
	HOOP /= 2
	HOOP = HOOP[2]
	route = '../data/heat4.route.npy'
	routes = np.load(route)
	
	team = routes[-1,:,1]
	coord = np.array( map(np.vstack, routes[...,0]))
	m, n, _ = coord.shape
	coord = movingaverage(coord.reshape(-1, 2*n), 10).reshape(m,n,2)
	
	polar = cart2polar(coord.reshape(-1, 2), HOOP).reshape(m,n,2)
	defScore = defenceScore(polar, team)
	pScore = pickScore(polar, team)

	
	sampleData = pickVector(polar,team)
	
	sampleData = np.hstack((sampleData, defVector(polar,team), coord.reshape(-1, 8)))
	#sampleData = defScore[0,:]
	#import ipdb; ipdb.set_trace()
	grammar = PR_tactic4()
	#grammar = pickGrammar()
	#grammar = defGrammar()
	#grammar = blockGrammar()
	model = PCFG(grammar)
	#import ipdb; ipdb.set_trace()
	lik, tree = model.decode(sampleData)
	decodes = []
	for x in tree.BFS():
		#if x[1][0] == x[1][1]:
		decodes.append(x)
	for x in sorted(decodes, key=lambda x:x[1][0]):
		print x

def heat1():
	from template import Pts4 as HOOP
	HOOP /= 2
	HOOP = HOOP[0]

	route = '../data/heat1.5.route.npy'
	routes = np.load(route)
	
	team = routes[-1,:,1]
	coord = np.array( map(np.vstack, routes[...,0]))
	m, n, _ = coord.shape
	coord = movingaverage(coord.reshape(-1, 2*n), 10).reshape(m,n,2)
	
	polar = cart2polar(coord.reshape(-1, 2), HOOP).reshape(m,n,2)
	defScore = defenceScore(polar, team)
	pScore = pickScore(polar, team)

	
	sampleData = pickVector(polar,team)
	
	sampleData = np.hstack((sampleData, defVector(polar,team), coord.reshape(-1, 8)))
	#sampleData = defScore[0,:]
	#import ipdb; ipdb.set_trace()
	grammar = PR_tactic1()
	#grammar = pickGrammar()
	#grammar = defGrammar()
	#grammar = blockGrammar()
	model = PCFG(grammar)
	#import ipdb; ipdb.set_trace()
	lik, tree = model.decode(sampleData)
	decodes = []
	for x in tree.BFS():
		#if x[1][0] == x[1][1]:
		decodes.append(x)
	for x in sorted(decodes, key=lambda x:x[1][0]):
		print x

def heat3():
	
	from template import Pts4 as HOOP
	HOOP /= 2
	HOOP = HOOP[0]

	route = '../data/heat3.route.npy'
	routes = np.load(route)
	
	team = routes[-1,:,1]
	coord = np.array( map(np.vstack, routes[...,0]))
	m, n, _ = coord.shape
	coord = movingaverage(coord.reshape(-1, 2*n), 10).reshape(m,n,2)
	
	polar = cart2polar(coord.reshape(-1, 2), HOOP).reshape(m,n,2)
	defScore = defenceScore(polar, team)
	pScore = pickScore(polar, team)

	
	sampleData = pickVector(polar,team)
	
	sampleData = np.hstack((sampleData, defVector(polar,team), coord.reshape(-1, 8)))
	#sampleData = defScore[0,:]
	#import ipdb; ipdb.set_trace()
	grammar = PR_tactic3()
	#grammar = pickGrammar()
	#grammar = defGrammar()
	#grammar = blockGrammar()
	model = PCFG(grammar)
	#import ipdb; ipdb.set_trace()
	lik, tree = model.decode(sampleData)
	decodes = []
	for x in tree.BFS():
		#if x[1][0] == x[1][1]:
		decodes.append(x)
	for x in sorted(decodes, key=lambda x:x[1][0]):
		print x

def heat7():
	from template import Pts4 as HOOP
	HOOP /= 2
	HOOP = HOOP[0]

	route = '../data/heat7.route.npy'
	routes = np.load(route)
	
	team = routes[-1,:,1]
	coord = np.array( map(np.vstack, routes[...,0]))
	m, n, _ = coord.shape
	coord = movingaverage(coord.reshape(-1, 2*n), 10).reshape(m,n,2)
	
	polar = cart2polar(coord.reshape(-1, 2), HOOP).reshape(m,n,2)
	defScore = defenceScore(polar, team)
	pScore = pickScore(polar, team)
	import ipdb; ipdb.set_trace()
	
	sampleData = pickVector(polar,team)
	
	sampleData = np.hstack((sampleData, defVector(polar,team), coord.reshape(-1, 8)))
	#sampleData = defScore[0,:]
	#import ipdb; ipdb.set_trace()
	grammar = PR_tactic7()
	#grammar = pickGrammar()
	#grammar = defGrammar()
	#grammar = blockGrammar()
	model = PCFG(grammar)
	#import ipdb; ipdb.set_trace()
	lik, tree = model.decode(sampleData)
	decodes = []
	for x in tree.BFS():
		#if x[1][0] == x[1][1]:
		decodes.append(x)
	for x in sorted(decodes, key=lambda x:x[1][0]):
		print x
	import ipdb; ipdb.set_trace()

def prGrammar(alpha=100, beta=-50, gamma=-200, theta=-500):
	S = Nonterminal('S')
	PR = Nonterminal('PR')
	PICK = Nonterminal('PICK')
	Pick = Nonterminal('Pick')
	Pick0 = Nonterminal('Pick0')
	Pick1 = Nonterminal('Pick1')
	pick = Nonterminal('pick')
	pick0 = Nonterminal('pick0')
	pick1 = Nonterminal('pick1')
	NULL = Nonterminal('NULL')
	Block = Nonterminal('Block')
	Defence = Nonterminal('Defence')

	gBlock = mixture.GMM(n_components=1)
	gBlock.means_ = np.array([[alpha]])
	gBlock.covars_ = np.array([[1]])

	block = Terminal('block', None, None, cb=lambda x:blockGood(x, range(12,20), gBlock))
	nonBlock = Terminal('nonBlock', None, None, cb=lambda x:blockBad(x, range(12,20), gBlock, beta))

	gPick = mixture.GMM(n_components=2)
	gPick.means_ = np.array([[100, .3], [100, -.3]])
	gPick.covars_ = np.array([[50, 1e-3], [50, 1e-3]])
	gPick.weights_ = np.array([.5, .5])
	Pick0 = Terminal('picker0', None, None, cb=lambda x:positive(x, [0,2], gPick))
	Pick1 = Terminal('picker1', None, None, cb=lambda x:positive(x, [1,3], gPick))
	Null = Terminal('N/A', None, None, cb=lambda x:negative(x, [0,1,2,3], gPick, gamma))

	gDef = mixture.GMM(n_components=1)
	gDef.means_ = np.array([[-60, 0]])
	gDef.covars_ = np.array([[10, 1e-2]])
	
	defGood = Terminal('defence', None, None, cb=lambda x:defenceGood(x, range(4,12), gDef))
	defBad = Terminal('defence miss', None, None, cb=lambda x:defenceBad(x, range(4,12), gDef, theta))

	prods = [ ]
	prods.append(Production(S, [NULL, PR], prob=1.))
	prods.append(Production(PR, [PICK, Defence], prob=1.))
	prods.append(Production(PICK, [Pick0, Block], prob=.5))
	prods.append(Production(PICK, [Pick1, Block], prob=.5))
	prods.append(Production(Defence, [Defence, defBad], prob=.8))
	prods.append(Production(Defence, [defBad, defBad], prob=.2))
	prods.append(Production(Pick0, [pick0, Pick], prob=.8))
	prods.append(Production(Pick0, [pick0, pick0], prob=.2))
	prods.append(Production(Pick1, [pick1, Pick], prob=.8))
	prods.append(Production(Pick1, [pick1, pick1], prob=.2))
	prods.append(Production(pick0, [Pick0, Pick0], prob=1.))
	prods.append(Production(pick1, [Pick1, Pick1], prob=1.))
	prods.append(Production(Block, [block, Block], prob=.8))
	prods.append(Production(Block, [block, block], prob=.2))
	prods.append(Production(NULL, [Null, NULL], prob=.8))
	prods.append(Production(NULL, [Null, Null], prob=.2))

	return Grammar(S, prods)

def recognition(args):
	from template import Pts4 as HOOP
	HOOP /= 2
	HOOP = HOOP[0]
	routes = np.load(args['route'])
	team = routes[-1,:,1]
	coord = np.array( map(np.vstack, routes[...,0]))
	m, n, _ = coord.shape
	coord = movingaverage(coord.reshape(-1, 2*n), 10).reshape(m,n,2)
	polar = cart2polar(coord.reshape(-1, 2), HOOP).reshape(m,n,2)

	sampleData = np.hstack((pickVector(polar,team), defVector(polar,team), coord.reshape(-1, 8)))
	grammar = prGrammar()
	model = PCFG(grammar)
	lik, tree = model.decode(sampleData)
	decodes = []
	for x in tree.BFS():
		#if x[1][0] == x[1][1]:
		decodes.append(x)
	for x in sorted(decodes, key=lambda x:x[1][0]):
		print x

def main():
	
	parser = argparse.ArgumentParser(description="detect trajectory")
	parser.add_argument('-s', '--src', help='case folder name', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/':
		args['src'] = args['src'] + "/"
	args['route'] = '%sroute.npy'%args['src']
	
	recognition(args)

if __name__ == '__main__':
	main()