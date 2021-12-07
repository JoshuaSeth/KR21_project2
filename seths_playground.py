'''File for testing implementations of functions and designing the functions before throwing them in the BNReasoner class.'''
from BayesNet import BayesNet
import pandas as pd
import copy
from BNReasoner import BNReasoner

BN = BayesNet()
BN.load_from_bifxml('testing/dog_problem.BIFXML')
BR = BNReasoner(BN)


m = BR.get_marginal_distribution(
    ["hear-bark", "dog-out"], ["family-out"])
print(m)
