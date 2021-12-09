import networkx as nx
from typing import Union
from xml.etree.ElementTree import TreeBuilder

from numpy import multiply
from BayesNet import BayesNet
import copy
import pandas as pd
pd.options.mode.chained_assignment = None  # disable bs warnings of Pandas


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
        """
        Given 2 probability tables multiplies them and returns the multiplied CPT. Example usage:
        cpt_1 = BN.get_cpt("hear-bark")
        cpt_2 = BN.get_cpt("dog-out")
        factor_product = BR.multiply_cpts(cpt_1, cpt_2)
        """
        # 1. get variables that is in 2nd cpt and not in 1st
        cpt_1_no_p = list(cpt_1)[:-1]
        vars_to_add = [col for col in list(
            cpt_2) if col not in cpt_1_no_p]

        # 2. Construct new CPT
        new_cpt_cols = cpt_1_no_p + vars_to_add
        index_cols_of_first_cpt = len(cpt_1_no_p)
        new_cpt_len = pow(2, len(new_cpt_cols)-1)
        new_cpt = pd.DataFrame(columns=new_cpt_cols,
                               index=range(new_cpt_len), dtype=object)

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
        """
        Returns the conditional probability table for variables in Q with the variables in E marginalized out.
        Q: list of variables for which you want a probability table.
        E: list of variables for which you want the marginalized distribution (opposite of marginalizing out).

        Example usage: 
        m = BR.get_marginal_distribution(["hear-bark", "dog-out"], ["family-out"])
        """
        # 1. multiply CPTs for different variables in Q to 1 big CPT
        # Get cpts for vars
        cpts = [self.bn.get_cpt(var) for var in Q]
        # Multiply them into 1 big cpt
        multiplied_cpt = cpts[0]
        for i in range(1, len(cpts)):
            cpt = cpts[i]
            multiplied_cpt = self.multiply_cpts(multiplied_cpt, cpt)

        # 2. marginalize out variables NOT in E
        for var in list(multiplied_cpt):
            if var not in E and var != "p":
                multiplied_cpt.drop(var, 1, inplace=True)
            # Sum up p for rows that are the same
            multiplied_cpt = multiplied_cpt.groupby(
                list(multiplied_cpt)[:-1]).sum().reset_index()

        return multiplied_cpt

    def pruner(self, Q, E):
        '''Returns pruned network for given variables Q and evidence E'''
        # deleting leaf nodes
        variables = self.bn.get_all_variables()
        for variable in variables:
            # if variable is not part of the selected variables ...
            if variable not in Q and variable not in E:
                children = self.bn.get_children([variable])
                # ... and has no children, then delete it
                if not children:
                    self.bn.del_var(variable)

        # deleting outgoing edges from E
        for evidence in E:
            children = self.bn.get_children([evidence])
            for child in children:
                self.bn.del_edge((evidence, child))

        return self.bn

    def get_all_paths(self, start_node, end_node):
        """
        Returns all paths between nodes
        """
        temp_network = copy.deepcopy(self.bn.structure)
        for edge in temp_network.edges:
            temp_network.add_edge(edge[1], edge[0])
        return nx.all_simple_paths(temp_network, source=start_node, target=end_node)

    def triple_active(self, nodes, evidence):
        for node in nodes:
            # 1. Determine the relationships
            other_nodes = [o_node for o_node in nodes if o_node != node]
            children = self.bn.get_children([node])
            parents = self.bn.get_parents([node])
            descendants = nx.descendants(self.bn.structure, node)
            ancestors = nx.ancestors(self.bn.structure, node)

            # 2. Find out which node is the middle node if causal relationship
            middle_node = "None yet"
            for alt_node in nodes:
                other_nodes_2 = [
                    o_node for o_node in nodes if o_node != alt_node]
                if (other_nodes_2[0] in self.bn.get_parents([alt_node]) and other_nodes_2[1] in self.bn.get_children([alt_node])) or (other_nodes_2[1] in self.bn.get_parents([alt_node]) and other_nodes_2[0] in self.bn.get_children([alt_node])):
                    middle_node = alt_node

            # 3. Check the 4 rules, x->y->z, x<-y<-z, x<-y->z, x->y<-z
            if set(other_nodes).issubset(parents) and node in evidence:  # V-structure
                return True
            if set(other_nodes).issubset(children) and node not in evidence:  # COmmon cause
                return True
            if not set(other_nodes).issubset(children) and set(other_nodes).issubset(descendants):  # Causal
                if middle_node not in evidence:
                    return True
            if not set(other_nodes).issubset(parents) and set(other_nodes).issubset(ancestors) and node not in evidence:  # Inverse-causal
                if middle_node not in evidence:
                    return True
        return False  # If none of the rules made the triple active the triple is false

    def d_separation_alt(self, var_1, var_2, evidence):
        """
        Given two variables and evidence returns if it is garantued that they are independent. 
        False means the variables are NOT garantued to independent. True means they are independent. 

        Example usage:
        var_1, var_2, evidence = "bowel-problem", "light-on", ["dog-out"]
        print(BR.d-separation_alt(var_1, var_2, evidence))
        """
        for path in self.get_all_paths(var_1, var_2):
            active_path = True
            triples = [[path[i], path[i+1], path[i+2]]
                       for i in range(len(path)-2)]
            for triple in triples:
                # Single inactive triple makes whole path inactive
                if not self.triple_active(triple, evidence):
                    active_path = False
            if active_path:
                return False  # indepence NOT garantued if any path active
        return True  # Indpendence garantued if no path active

    def get_joint_probability_distribution(self):
        '''Returns full joint probability distribution table when applied to a 
        Bayesian Network'''
        all_variables = self.bn.get_all_variables()
        final_table = self.bn.get_cpt(all_variables[0])

        # multiplies all CPT to get a JPD
        for i in range(1, len(all_variables)):
            table_i = self.bn.get_cpt(all_variables[i])
            final_table = self.multiply_cpts(final_table, table_i)

        return final_table

    def summing_out(self, sum_out_variables):
        '''Takes set of variables that needs to be summed out as an input and 
        returns joint probability distribution table with given variables 
        eliminated when applied to a Bayesian Network'''
        # get full JPD
        JPD = self.get_joint_probability_distribution()

        # delete columns of variables that need to be summed out
        JPD = JPD.drop(columns=list(sum_out_variables))

        # sum up p values of remaining rows if the are similar
        remaining_columns = list(
            set(self.bn.get_all_variables()) - set(sum_out_variables))
        PD_new = JPD.groupby(remaining_columns).aggregate({'p': 'sum'})

        return PD_new

    def min_degree_ordening(self, X: list) -> dict:
        all_degrees = []
        all_degrees = [self.number_of_edges(e) for e in X]
        dict_of_degrees = dict(zip(X, all_degrees))  # unsorted
        # lowest values first, for easy of use, can chance to list
        dict_of_degrees_sorted = dict(
            sorted(dict_of_degrees.items(), key=lambda item: item[1]))
        return dict_of_degrees_sorted

    def number_of_edges(self, X: str) -> int:
        length = len(int_graph[X])
        return length
        # wrote this but.. can use network.number_of_edges(u, v) where u and v are nodes to count between, empty input = all edges

    def get_all_triangles(self, network):
        all_linked_nodes = list(nx.enumerate_all_cliques(network))
        all_triangles = [c for c in all_linked_nodes if len(c) == 3]
        # was double list brackets, might be a problem later if multiple triangles?
        return all_triangles[0]

    # gets all nodes with a single edge in a given interaction network
    def nodes_with_1_edge(self, network):
        single_connection_node = []
        for i in BN.get_all_variables():
            all_linked_nodes = network[i]
            if len(all_linked_nodes) == 1:
                single_connection_node.append(i)
        return single_connection_node

    def get_only_triangle_node(self, int_graph, all_nodes: list) -> str:
        for elements in all_nodes:
            connections = list(int_graph[elements])
            # checking which connections are only within the triangle
            lst_check = all(elem in all_nodes for elem in connections)
            if lst_check == True:
                var_to_remove = elements
        return var_to_remove

    def min_fill_ordening(self, network) -> dict:
        '''
        current idea to check the interaction graph for triangles, 
        because we know that if there is a triangle, 
        we can remove the node which only has connections to other nodes within this triangle.

        another 'free' deletion is the deletion of nodes which have only 1 connection, because deleting them never causes an added edge.
        '''
        nodes_with_value_0 = []
        all_triangles = self.get_all_triangles(network)  # get all triangles
        # get all nodes from triangles which can be deleted without adding edge
        triangle_node_to_remove = self.get_only_triangle_node(
            network, all_triangles)
        # add triangle nodes to value 0 list
        nodes_with_value_0.append(triangle_node_to_remove)
        single_edge_nodes = self.nodes_with_1_edge(
            network)  # get all single edge nodes
        for elements in single_edge_nodes:
            nodes_with_value_0.append(elements)
        lst_0 = [0] * len(nodes_with_value_0)
        min_fill_dict = dict(zip(nodes_with_value_0, lst_0))
        print(min_fill_dict)


# test summing-out
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
print(bn_grass.summing_out(('Slippery Road?', 'Sprinkler?', 'Rain?')))

# test get JPD
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
print(bn_grass.get_joint_probability_distribution())
print(bn_grass.get_joint_probability_distribution().sum())
'''

# test 2 multiplying factors
'''
BN = BayesNet()
BN.load_from_bifxml('testing/lecture_example.BIFXML')
BR = BNReasoner(BN)
cpt_1, cpt_2 = BN.get_cpt('Winter?'), BN.get_cpt('Sprinkler?')
print(BR.multiply_cpts(cpt_1, cpt_2))
'''

# test multiplying factors
'''
BN = BNReasoner('testing/dog_problem.BIFXML')
cpt_1 = BN.bn.get_cpt("hear-bark")
cpt_2 = BN.bn.get_cpt("dog-out")
print('cpt_1:', cpt_1, 'cpt_2:', cpt_2)
factor_product = BN.multiply_cpts(cpt_1, cpt_2)
print('factor_product:', factor_product) 
'''

# test pruner
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
bn_grass.bn.draw_structure()
bn_grass.pruner({'Winter?', 'Wet Grass?'},{'Sprinkler?'})
bn_grass.bn.draw_structure()
'''

# test d-separation
'''
reasoner = BNReasoner('testing/dog_problem.BIFXML')
BN = BayesNet()
network = BN.load_from_bifxml('testing/dog_problem.BIFXML')
BN.draw_structure()
test = reasoner.d_separation(network, 'family-out', 'hear-bark', ['dog-out'])
print(test)

# test = reasoner.d_separation(network, 'dog-out', 'light-on', ['dog-out'])
# print(test)
'''
