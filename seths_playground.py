'''File for testing implementations of functions and designing the functions before throwing them in the BNReasoner class.'''
from networkx.algorithms.dag import ancestors
from BayesNet import BayesNet
import pandas as pd
import copy
from BNReasoner import BNReasoner
import networkx as nx

BN = BayesNet()
BN.load_from_bifxml('USE_CASE_NETWORK_DIAGNOSIS.BIFXML')
BR = BNReasoner(BN)


# var_1, var_2, evidence = "bowel-problem", "light-on", ["dog-out"]


cpt_1 = BR.bn.get_cpt("Chronic_lung_disease")
cpt_2 = BR.bn.get_cpt("Cardiovascular_disease")
# print(BR.multiply_cpts_extensive(cpt_1, cpt_2))
# print(cpt_1)
# print(cpt_2)

p = BR.get_marginal_distribution(
    ["Cardiovascular_disease"], {"Tightnessinchest": True, "Highbloodpressure": True, "Leftventricularhypertrophy": False})

print(p)


# cpt_3 = BR.bn.get_cpt("Wet Grass?")
# cpt_1 = cpt_1.drop([0])
# cpt_2 = cpt_2.drop([0, 1])
# cpt_3 = cpt_3.drop([1, 3, 5, 7])

# # print(cpt_1)
# print(cpt_2)
# print(cpt_3)

# print("result")
