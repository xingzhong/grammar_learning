from BiCluster import *
from nltk.corpus import brown

sample = brown.sents()
sample2 = sample[:1000]
print learnGrammar(sample2)