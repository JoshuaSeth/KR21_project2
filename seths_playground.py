'''File for testing implementations of functions and designing the functions before throwing them in the BNReasoner class.'''
from networkx.algorithms.dag import ancestors
from BayesNet import BayesNet
import pandas as pd
import copy
from BNReasoner import BNReasoner
import networkx as nx

BN = BayesNet()
BN.load_from_bifxml('testing/dog_problem.BIFXML')
BR = BNReasoner(BN)


var_1, var_2, evidence = "bowel-problem", "light-on", ["dog-out"]

print(BR.d_separation_alt(var_1, var_2, evidence))
