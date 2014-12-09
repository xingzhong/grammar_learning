import numpy as np
np.set_printoptions(precision=3)
from pcfg import sample, Dist, sample2
from cyk import cyk
from sklearn import hmm
import matplotlib.pyplot as plt

def markov():
	startprob = np.array([1.0, 0, 0])
	transmat = np.array([[0.2, 0.8, 0.0], [1/3.0, 1/3.0, 1/3.0], [1.0, 0, 0]])
	means = np.array([[2.0], [-2.0], [0.0]])
	covars = np.array([[0.1], [0.1], [0.1]])
	model = hmm.GaussianHMM(3, 'diag', startprob, transmat)
	model.means_ = means
	model.covars_ = covars
	return model

def markovLearning(s):
	model = hmm.GaussianHMM(3, 'diag')
	model.fit([s])
	print model.means_
	print model.covars_
	print model.transmat_
	return model

def grammar():
	A = Dist(np.array([2.0]), np.array([0.1]))
	B = Dist(np.array([-2.0]), np.array([0.1]))
	C = Dist(np.array([0.0]), np.array([0.1]))

	G = [ ('NT_1', A, None, 1.0), ('NT_2', B, None, 1.0), ("NT_3", C, None, 1.0), 
			("NT_4", "NT_1", "NT_2", 1.0), # nt4->a b 
			("NT_5", "NT_2", "NT_3", 1.0), # nt5->b c
			("NT_6", "NT_4", "NT_6", 0.5), 
			("NT_6", "NT_1", "NT_2", 0.5), # nt6->nt4 nt6 | nt1 nt2
			("NT_7", "NT_6", "NT_5", 1.0), # nt7->nt6 nt5
			("NT_8", "NT_7", "NT_1", 1.0), # nt8->nt7 nt1
			("NT_S", "NT_7", "NT_1", 0.5), # nts->nt8 nts | nt7 nt1
			("NT_S", "NT_8", "NT_S", 0.5)] 
	return G

def vis(s1,s2, markovLik1, markovLik2, cyklik1, cyklik2):
	figsize=(30,8)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(2, 1, 1)
	ax.plot(s1, '--o')
	ax.set_title("Markov = %.4f \n CYK = %.4f"%(markovLik1, cyklik1))
	plt.grid()
	bx = fig.add_subplot(2, 1, 2)
	bx.plot(s2, '--o')
	bx.set_title("Markov = %.4f \n CYK = %.4f"%(markovLik2, cyklik2))
	plt.grid()
	plt.show()

if __name__ == '__main__':
	_, s1 = sample()
	_, s2 = sample2()
	g = grammar()
	m = markov()
	markovLik1, _ = m.decode(s1)
	markovLik2, _ = m.decode(s2)
	cyklik1 = cyk(s1, g)
	cyklik2 = cyk(s2, g)
	vis(s1, s2, markovLik1, markovLik2, cyklik1, cyklik2 )
	#markovLearning(s1)
	#markovLearning(s2)
