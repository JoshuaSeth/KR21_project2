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
            
    def get_z_parents(self, z):
        variables_to_check = copy(z)
        z_parents = set()
        while variables_to_check:
            var_to_check = self.structure.nodes[variables_to_check.pop()]
        for parent in var_to_check.parents:
            z_parents.add(parent)
        return z_parents

    def d_separation(self, x, y, z):
            z_parents = self.get_z_parents(z)
            nodes_to_visit = [(x, 'asc')]
            already_visited = set()

            while nodes_to_visit:
                node_name, up_or_down = nodes_to_visit.pop()
                node = self.structure.nodes[node_name]

                if (node_name, up_or_down) not in already_visited:
                    already_visited.add((node_name, up_or_down))

                    if node_name not in z and node_name == y:
                        return False

                    if up_or_down == 'asc' and node_name not in z:
                        for parent in self.structure.parents:
                            nodes_to_visit.append((parent,'asc'))
                        for child in self.structure.children:
                            nodes_to_visit.append((child, 'des'))

                    elif up_or_down == 'des':
                        if node_name not in z:
                            for child in node.children:
                                nodes_to_visit.append((child, 'des'))
                        if node_name in z or node_name in z_parents:
                            for parent in node.parents:
                                nodes_to_visit.append((parent, 'asc'))
            return True
    
reasoner = BNReasoner('testing/dog_problem.BIFXML')
reasoner.d_separation('light-on', 'hear-bark', ['family-out'])

            
    # TODO: This is where your methods should go
