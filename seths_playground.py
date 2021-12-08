'''File for testing implementations of functions and designing the functions before throwing them in the BNReasoner class.'''
from networkx.algorithms.dag import ancestors
from BayesNet import BayesNet
import pandas as pd
import copy
from BNReasoner import BNReasoner
import networkx as nx

# BN = BayesNet()
# BN.load_from_bifxml('testing/lecture_example.BIFXML')
# BR = BNReasoner(BN)

# cpt_1, cpt_2 = BN.get_cpt('Winter?'), BN.get_cpt('Sprinkler?')
# print(BR.multiply_cpts(cpt_1, cpt_2))

BN = BNReasoner('testing/dog_problem.BIFXML')
cpt_1 = BN.bn.get_cpt("hear-bark")
cpt_2 = BN.bn.get_cpt("dog-out")
print('cpt_1:', cpt_1, 'cpt_2:', cpt_2)
factor_product = BN.multiply_cpts(cpt_1, cpt_2)
print('factor_product:', factor_product)

var_1, var_2, evidence = "bowel-problem", "light-on", ["dog-out"]
