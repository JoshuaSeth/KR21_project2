from typing import Union
from BayesNet import BayesNet
import copy
import pandas as pd
import networkx as nx


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

    def get_all_paths(self, start_node, end_node):
        '''Returns all possible paths between '''
        return nx.all_simple_paths(self.bn.structure, start_node, end_node)

    def d_separation(self, network, x, y, z):
        # test
        z_parents = BN.get_parents(z)
        z_children = BN.get_children(z)
        print(z_children)
        print(z_parents)
        nodes_to_visit = [(x, 'asc')]
        already_visited = set()
        nodes = set(BN.get_all_variables())

        while nodes_to_visit:
            (node_name, up_or_down) = nodes_to_visit.pop()

            # if current visiting node is not already_visited, skip it
            if (node_name, up_or_down) not in already_visited:
                already_visited.add((node_name, up_or_down))

                if node_name not in z and node_name == y:  # if we reach the end, no d-separation
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

    def multiply_cpts(self, cpt_1, cpt_2):
        '''Given 2 probability tables multiplies them and returns the multiplied CPT. Example usage:
        \ncpt_1 = BN.get_cpt("hear-bark")
        \ncpt_2 = BN.get_cpt("dog-out")
        \nfactor_product = BR.multiply_cpts(cpt_1, cpt_2)'''
        # 1. get variables that is in 2nd cpt and not in 1st
        cpt_1_no_p = list(cpt_1)[:-1]
        vars_to_add = [col for col in list(
            cpt_2) if col not in cpt_1_no_p]

        # 2. Construct new CPT
        new_cpt_cols = cpt_1_no_p + vars_to_add
        index_cols_of_first_cpt = len(cpt_1_no_p)
        new_cpt_len = pow(2, len(new_cpt_cols)-1)
        new_cpt = pd.DataFrame(columns=new_cpt_cols, index=range(new_cpt_len))

        # 3. Fill in CPT with Trues and falses
        for i in range(len(new_cpt_cols)-1):
            rows_to_fill_in = pow(2, len(new_cpt_cols)-2-i)
            cur_bool = False
            for j in range(int(new_cpt_len/rows_to_fill_in)):
                start_i = j * rows_to_fill_in
                cur_bool = not cur_bool
                new_cpt[new_cpt_cols[i]][start_i:start_i +
                                         rows_to_fill_in] = cur_bool

        # 4. Get the rows in the current CPTs that correspond to values and multiply their p's
        for index, row in new_cpt.iterrows():
            cols = list(new_cpt)[:-1]
            p_1 = copy.deepcopy(cpt_1)
            p_2 = copy.deepcopy(cpt_2)

            index_1 = 0
            for col in cols:
                if col in list(cpt_1):
                    p_1 = p_1.loc[p_1[col] == row[index_1]]
                if col in list(cpt_2):
                    p_2 = p_2.loc[p_2[col] == row[index_1]]
                index_1 += 1
            result = float(p_1["p"].item()) * float(p_2["p"].item())
            new_cpt["p"][index] = result

        return new_cpt

    def get_marginal_distribution(self, Q, E):
        '''Returns the conditional probability table for variables in Q with the variables in E marginalized out. \n Q: list of variables for which you want a probability table. \n E: list of variables that need to be marginalized out. \n\n Example usage: \n m = BR.get_marginal_distribution(["hear-bark", "dog-out"], ["family-out"])'''
        # 1. multiply CPTs for different variables in Q to 1 big CPT
        # Get cpts for vars
        cpts = [self.bn.get_cpt(var) for var in Q]
        # Multiply them into 1 big cpt
        multiplied_cpt = cpts[0]
        for i in range(1, len(cpts)):
            cpt = cpts[i]
            multiplied_cpt = self.multiply_cpts(multiplied_cpt, cpt)

        # 2. marginalize out variables in E
        for var in E:
            multiplied_cpt.drop(var, 1, inplace=True)
        # Sum up p for rows that are the same
        multiplied_cpt = multiplied_cpt.groupby(
            list(multiplied_cpt)[:-1]).sum().reset_index()

        return multiplied_cpt


reasoner = BNReasoner('testing/dog_problem.BIFXML')
BN = BayesNet()
network = BN.load_from_bifxml('testing/dog_problem.BIFXML')
BN.draw_structure()
test = reasoner.d_separation(network, 'family-out', 'hear-bark', ['dog-out'])
print(test)

# TODO: This is where your methods should go
