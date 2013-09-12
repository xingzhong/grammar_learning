from BiCluster import *
from nltk.corpus import brown

sample = brown.sents()
sample2 = sample[:10000]
print learnGrammar(sample2)
