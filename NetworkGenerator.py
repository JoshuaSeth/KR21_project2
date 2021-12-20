import math
import pandas as pd
import random

from BayesNet import BayesNet

class NetworkGenerator:
    def __init__(self):
        pass


    def generate_network(self, network_size: int) -> BayesNet:
        bn = BayesNet()

        variables = []
        edges = []
        cpts = {}

        for i in range(network_size):
            # create variable
            var = str(i)

            # choose parents
            n_parents = min(len(variables), random.randint(1, 16), math.ceil(math.sqrt(len(variables))))
            parents = [variables[j] for j in random.sample(range(len(variables)), n_parents)]
            new_edges = [(var, parent) for parent in parents]
            
            # create cpt
            cpt = self.generate_cpt(var, parents)

            # store variable, edges and cpt
            variables.append(var)
            edges.extend(new_edges)
            cpts[var] = cpt

        # create network
        bn.create_bn(variables, edges, cpts)

        return bn

    
    def generate_cpt(self, node: str, parents: list) -> pd.DataFrame:
        """
        Given a node ant a list of its parents, returns a CPT with random probabilities
        """
        
        n_parents = len(parents)

        cpt_list = []
        for i in range(2**n_parents):
            tf_values = [True if i & (1 << j) else False for j in range(n_parents)]
            
            random_prob = random.random()
            cpt_list.append(tf_values + [True, random_prob])
            cpt_list.append(tf_values + [False, 1 - random_prob])

        cpt = pd.DataFrame(cpt_list, columns=parents + [node, 'p'])
        return cpt

                

if __name__ == '__main__':
    ng = NetworkGenerator()

    bn = ng.generate_network(100)
    bn.draw_structure()
