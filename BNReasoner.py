from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
            
    def d_separation(self, network, x, y, z):
            z_parents = BN.get_parents(z)
            z_children = BN.get_children(z)
            print(z_children)
            print(z_parents)
            nodes_to_visit = [(x, 'asc')]
            already_visited = set()
            nodes = set(BN.get_all_variables())

            while nodes_to_visit:
                (node_name, up_or_down) = nodes_to_visit.pop()

                if (node_name, up_or_down) not in already_visited: # if current visiting node is not already_visited, skip it
                    already_visited.add((node_name, up_or_down))

                    if node_name not in z and node_name == y: # if we reach the end, no d-separation
                        return False

                    if up_or_down == 'asc' and node_name not in z:
                        for parent in z_parents:
                            nodes_to_visit.append((parent, 'asc'))
                        for child in z_children:
                            nodes_to_visit.append((child, 'des'))
                    elif up_or_down == 'des':
                        if node_name not in z:
                            for child in z_children:
                                nodes_to_visit.append((child, 'des'))
                        if node_name in z or node_name in z_parents:
                            for parents in z_parents:
                                nodes_to_visit.append((parents, 'asc'))
            return True
        
reasoner = BNReasoner('testing/dog_problem.BIFXML')
BN = BayesNet()
network = BN.load_from_bifxml('testing/dog_problem.BIFXML')
BN.draw_structure()
test= reasoner.d_separation(network, 'family-out', 'hear-bark', ['dog-out'])
print(test)

            
    # TODO: This is where your methods should go
