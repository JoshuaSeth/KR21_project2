from BayesNet import BayesNet
import pandas as pd
import copy
from BNReasoner import BNReasoner

BN = BayesNet()
BN.load_from_bifxml('testing/dog_problem.BIFXML')
BR = BNReasoner(BN)


def get_marginal_distribution(Q, E):
    '''Returns the conditional probability table for variables in Q with the variables in E marginalized out.'''
    # 1. multiply CPTs for different variables in Q to 1 big CPT
    # 2. marginalize out variables in E

    # 1.1 Get CPTs for variables in Q
    cpts_for_Q = []
    for key, value in BN.get_all_cpts():
        if key in Q:
            cpts_for_Q.append(value)

    # 1.2 Multiply them into 1 big CPT


cpt_1 = BN.get_cpt("hear-bark")
cpt_2 = BN.get_cpt("dog-out")
factor_product = BR.multiply_cpts(cpt_1, cpt_2)

print(t)
