from itertools import izip_longest
from pprint import pprint

test = '1 2 3 4 5 6 7 8'
tokens = test.split()
def generate(tokens):
  for i in range(len(tokens)+1):
    left  = tokens[:i]
    right = tokens[i:][::-1]
    pairs = [None]*len(right) + left + [None]*len(right)
    for j in range(len(pairs)):
      right_j = [None]*j + right
      candidate = filter( any, izip_longest(pairs, right_j))
      yield (candidate)

for ind, seq in enumerate( generate(tokens) ):
  print ind, seq

