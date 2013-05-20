# -*- coding: utf-8 -*-

from hmm import Model
from scipy.stats import norm
import hmm
import numpy as np

states = ('A', 'B', 'C')
symbols = ('a', 'b', 'c')

start_prob = {
    'A' : 1.0/3,
    'B' : 1.0/3,
    'C' : 1.0/3
}

trans_prob = {
    'A': { 'A' : 0.4, 'B' : 0.3, 'C' : 0.3 },
    'B': { 'A' : 0.4, 'B' : 0.3, 'C' : 0.3 },
    'C': { 'A' : 0.4, 'B' : 0.3, 'C' : 0.3 }
}

emit_prob = {
    'A': { 'mean' : [1, 1], "cov" : np.eye(2)},
    'B': { 'mean' : [0, 0], "cov" : np.eye(2)},
    'C': { 'mean' : [-1, -1], "cov" : np.eye(2)}
}

sequence = 4*np.random.rand(10, 2)-2
model = Model(states, symbols, start_prob, trans_prob, emit_prob)
print sequence
print model.evaluate(sequence)
print model.decode(sequence)
