import math
import pandas as pd
import random

from BayesNet import BayesNet

class NetworkGenerator:
    def __init__(self):
        pass

    def generate_network(self, network_size: int) -> BayesNet:
        """
        Generates a random BayesNet with the given network_size.
        """
        bn = BayesNet()

        nodes = []
        for i in range(network_size):
            n_parents = min(len(nodes), random.randint(1, math.ceil(math.sqrt(network_size))))
            
            parents = [node[0] for node in random.sample(nodes, n_parents)]
            nodes.append((i, parents))

        for node in nodes:
            var = node[0]
            parents = node[1]
            cpt = self.generate_cpt(var, parents)
            bn.add_var(var, cpt)
            for parent in parents:
                bn.add_edge((parent, var))

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

    for i in range(5):
        bn = ng.generate_network(10)
        bn.draw_structure()
