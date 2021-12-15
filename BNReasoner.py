import numpy as np
import random
import pandas as pd
import networkx as nx
from typing import Union
from xml.etree.ElementTree import TreeBuilder

from numpy import multiply
from BayesNet import BayesNet
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
        # print(z_children)
        # print(z_parents)
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

    @staticmethod
    def __multiply_cpts__(cpt1: pd.DataFrame, cpt2: pd.DataFrame) -> pd.DataFrame:
        """
        Multiplies two CPTs and returns the combined one
        :param cpt1: the first CPT
        :param cpt2: the seconds CPT
        :return combined CPT
        """
        result = cpt1.copy(deep=True)
        cpt1_columns = set(cpt1.columns) - {'p'}
        cpt2_columns = set(cpt2.columns) - {'p'}

        diff = cpt2_columns.difference(cpt1_columns)

        for d in diff:
            insert_idx = len(cpt1_columns) - 1
            if {True, False} == set(cpt2[d]):
                old = copy.deepcopy(result)
                result.insert(insert_idx, d, True)
                result = pd.concat([result, old]).fillna(False)
            else:
                for tv in [True, False]:
                    result.insert(insert_idx, d, tv)
            result = result.sort_values(by=list(result.columns)).reset_index(drop=True)

        for idx_result_row, result_row in result.iterrows():
            for _, cpt2_row in cpt2.iterrows():
                if result_row[cpt2_columns].equals(cpt2_row[cpt2_columns]):
                    result.at[idx_result_row, 'p'] *= cpt2_row['p']

        return result

    def is_unique(self, s):
        '''Quick check if all values in df are equal'''
        if not isinstance(s, pd.DataFrame):
            s.to_frame()  # If we got a series object transform it to DF
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    def multiply_cpts(self, cpt_1, cpt_2):
        """
        Given 2 probability tables multiplies them and returns the multiplied CPT. Example usage:
        cpt_1 = BN.get_cpt("hear-bark")
        cpt_2 = BN.get_cpt("dog-out")
        factor_product = BR.multiply_cpts(cpt_1, cpt_2)
        """
        # 0. Convert to df's if necessary
        if not isinstance(cpt_1, pd.DataFrame):
            cpt_1.to_frame()
        if not isinstance(cpt_2, pd.DataFrame):
            cpt_2.to_frame()

        # Reset their indices so we don't have weird errors
        if pd.Index(np.arange(0, len(cpt_1))).equals(cpt_1.index):
            cpt_1.reset_index()
        if pd.Index(np.arange(0, len(cpt_2))).equals(cpt_2.index):
            cpt_2.reset_index()

        # If there is an index column delete it since it means there is a double index
        if "index" in list(cpt_1):
            cpt_1.drop("index", 1, inplace=True)
        if "index" in list(cpt_2):
            cpt_2.drop("index", 1, inplace=True)

        # 1. get variables that is in 2nd cpt and not in 1st
        cpt_1_no_p = list(cpt_1)[:-1]
        vars_to_add = [col for col in list(
            cpt_2) if col not in cpt_1_no_p]

        # If columns consist of one single equal value the new cpt must be shorter
        singular_cols = [col for col in list(
            cpt_1_no_p) if self.is_unique(cpt_1[col]) and col != 'p']
        singular_cols += [col for col in list(
            cpt_2[:-1]) if self.is_unique(cpt_2[col]) and col not in singular_cols and col != 'p']
        discount = len(singular_cols)

        # Remebr the only value these cols had: False or True
        singular_vals = [cpt_1[col].iloc[0] for col in list(
            cpt_1_no_p) if self.is_unique(cpt_1[col]) and col != 'p']
        singular_vals += [cpt_2[col].iloc[0] for col in list(
            cpt_2[:-1]) if self.is_unique(cpt_2[col]) and col != 'p']

        # print(singular_cols, singular_vals)
        # 2. Construct new CPT
        new_cpt_cols = cpt_1_no_p + vars_to_add
        new_cpt_len = pow(2, len(new_cpt_cols)-1-discount)
        new_cpt = pd.DataFrame(columns=new_cpt_cols,
                               index=range(new_cpt_len), dtype=object)

        # 3. Fill in CPT with Trues and falses
        for i in range(len(new_cpt_cols)-1):
            # If this was a singular value column
            if new_cpt_cols[i] in singular_cols:
                new_cpt.loc[:, list(new_cpt_cols)[
                    i]] = singular_vals[singular_cols.index(new_cpt_cols[i])]
                continue
            rows_to_fill_in = pow(2, len(new_cpt_cols)-2-i)
            cur_bool = False
            for j in range(int(new_cpt_len/rows_to_fill_in)):
                start_i = j * rows_to_fill_in
                cur_bool = not cur_bool
                new_cpt[new_cpt_cols[i]][start_i:start_i +
                                         rows_to_fill_in] = cur_bool
        # print("filling in vals")
        # print(new_cpt)

        # 4. Get the rows in the current CPTs that correspond to values and multiply their p's
        for index, row in new_cpt.iterrows():
            cols = list(new_cpt)[: -1]
            p_1 = copy.deepcopy(cpt_1)
            p_2 = copy.deepcopy(cpt_2)

            index_1 = 0
            for col in cols:
                if col in list(cpt_1):
                    p_1 = p_1.loc[p_1[col] == row[index_1]]
                if col in list(cpt_2):
                    p_2 = p_2.loc[p_2[col] == row[index_1]]
                index_1 += 1
            # print(p_1)
            # print(p_2)
            result = float(p_1["p"].item()) * float(p_2["p"].item())
            new_cpt["p"][index] = result

        return new_cpt

    # NOTE: This must become boh a-priori and a-posteriori, currently it is a-posteriori
    # Posterioi is after evidence given, priori is distribution of single variable without evidence
    # I think I also need to integrate the whole multiplication chain i.e. Right now it just multiplies A and B when those ar ein Q but it needs to multiply the whole chain from the start until arriving at the variable for which we want the marginal distirbution. For example sometimes to get to C you have to do A x B|A x C|B before you arrive at the correct marginal distirbution. Right now this is just B|A x C|B. See video PGM 3 35:52 and 1:20:30
    # Variable must change Q is variables for which we want marginal distribution E is evidence
    # Currently it just multiplies tables and Q and marginalizes until ending up with E
    def get_marginal_distribution(self, Q, E):
        """
        Returns the conditional probability table for variables in Q with the variables in E marginalized out.
        Q: list of variables for which you want a marginal distribution.
        E: dict of variables with evidence. Leave empty if you want a-priori distribution

        Example usage:
        m = BR.get_marginal_distribution(
            ["hear-bark", "dog-out"], {"family-out":True})
        """
        # Alt Get vars in Q and multiply and sum out their chain
        results = []
        for var in Q:
            # get list of ancestors + var itself
            ancestors = list(nx.ancestors(
                self.bn.structure, var)) + [var]

            # multiply until arriving at this var
            current_table = self.bn.get_cpt(ancestors[0])
            for i in range(1, len(ancestors)):
                ancestor = ancestors[i]

                # And multiply with the next
                current_table = self.multiply_cpts(
                    current_table, self.bn.get_cpt(ancestor))
                results.append(current_table)

        # Then multiply those two final resulting vars in Q
        end = results[0]
        for j in range(1, len(results)):
            end = self.multiply_cpts(end, results[j])

        end = self.get_joint_probability_distribution()
        # end = self.bn.get_cpt(Q[0])
        # for i in range(1, len(Q)):
        #     end = self.multiply_cpts(end, self.bn.get_cpt(Q[i]))
        # print(end)
        # Marginalize out the evidence
        for col in list(end)[:-1]:
            # If E is empty this will simply be a-priori distribution
            if col not in list(E.keys()) and col not in Q:
                end.drop(col, 1, inplace=True)
                end = end.groupby(
                    list(end)[:-1]).aggregate({'p': 'sum'}).reset_index()
            # Else we will need to drop the rows contrary to evidence instead of whole variable
            if col in list(E.keys()) and col not in Q:
                end = end[end[col] == E[col]]  # rows contrary evidence
                # Only relevant rows still here so drop col
                end.drop(col, 1, inplace=True)
                end = end.groupby(list(end)[:-1]).aggregate(
                    {'p': 'sum'}).reset_index()  # Now group other cols (with only relevant p's)

        return end

    def pruner(self, Q, E):
        '''Returns pruned network for given variables Q and evidence E'''
        # create copy of network to work on
        network = copy.deepcopy(self.bn)

        # deleting leaf nodes
        variables = network.get_all_variables()
        for variable in variables:
            # if variable is not part of the selected variables ...
            if variable not in Q and variable not in E:
                children = network.get_children([variable])
                # ... and has no children, then delete it
                if not children:
                    network.del_var(variable)

        # deleting outgoing edges from E
        for evidence in E:
            children = network.get_children([evidence])
            for child in children:
                network.del_edge((evidence, child))

        return network

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
        # print(all_variables)
        final_table = self.bn.get_cpt(all_variables[0])
        # print(final_table)

        # multiplies all CPT to get a JPD
        for i in range(1, len(all_variables)):
            table_i = self.bn.get_cpt(all_variables[i])
            final_table = self.multiply_cpts(final_table, table_i)

        return final_table

    def JPD_and_summing_out(self, sum_out_variables):
        '''Takes set of variables that needs to be summed out as an input and
        returns joint probability distribution table with given variables
        eliminated when applied to a Bayesian Network'''
        # get full JPD
        JPD = self.get_joint_probability_distribution()

        # delete columns of variables that need to be summed out
        JPD = JPD.drop(columns=list(sum_out_variables))

        # sum up p values of remaining rows if they are similar
        remaining_columns = list(
            set(self.bn.get_all_variables()) - set(sum_out_variables))
        PD_new = JPD.groupby(remaining_columns).aggregate({'p': 'sum'})

        return PD_new

    def summing_out(self, cpt, sum_out_variables):
        '''Takes set of variables (given als list of strings) that needs to be
        summed out as an input and returns table with without given variables
        when applied to a Bayesian Network'''

        # delete columns of variables that need to be summed out
        cpt = cpt.drop(columns=sum_out_variables)

        # get the variables still present in the table
        remaining_variables = list(cpt.columns.values)[:-1]

        # sum up p values if rows are similar
        PD_new = cpt.groupby(remaining_variables).aggregate({'p': 'sum'})

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
        int_graph = BN.get_interaction_graph()
        length = len(int_graph[X])
        return length

    def min_fill_ordering(self, X: list) -> list:
        '''Input list of strings'''
        int_graph = BN.get_interaction_graph()
        num_edges = []
        for x in X:
            all_neighbors = list(int_graph.neighbors(x))
            # might need to change r if large number of connections?
            all_combinations_of_neighbors = list(
                itertools.combinations(all_neighbors, r=2))
            new_edges = [
                i for i in all_combinations_of_neighbors if i not in int_graph.edges]
            number_of_new_edges = len(new_edges)
            num_edges.append(number_of_new_edges)
        return dict(zip(X, num_edges))  # change to list/dict if necessary

    def JPD_and_maxing_out(self, max_out_variables):
        '''Takes set of variables that needs to be maxed out as an input and
        returns joint probability distribution table with given variables
        eliminated when applied to a Bayesian Network'''
        # get full JPD
        JPD = self.get_joint_probability_distribution()

        # delete columns of variables that need to be maxed out
        JPD = JPD.drop(columns=list(max_out_variables))

        # take max p value for remaining rows if they are similar
        remaining_columns = list(
            set(self.bn.get_all_variables()) - set(max_out_variables))
        PD_new = JPD.groupby(remaining_columns).aggregate({'p': 'max'})

        return PD_new

    def maxing_out(self, cpt, max_out_variables):
        '''Takes set of variables (given als list of strings) that needs to be
        maxed out as an input and returns table with without given variables
        when applied to a Bayesian Network'''

        # delete columns of variables that need to be maxed out
        cpt = cpt.drop(columns=max_out_variables)

        # get the variables still present in the table
        remaining_variables = list(cpt.columns.values)[:-1]

        # take max p value for remaining rows if they are similar
        PD_new = cpt.groupby(remaining_variables).aggregate({'p': 'max'})
        # print(PD_new)
        return PD_new

    def random_ordening(self, vars: list) -> list:
        """
        Returns a shuffled list of variables
        """
        random.shuffle(vars)
        return vars

    def condition(self, cpt: pd.DataFrame, evidence: list):
        """
        Given a CPT and evidence, returns a conditioned CPT
        """
        for (var, value) in evidence:
            if var in cpt.columns:
                cpt = cpt.loc[cpt[var] == value]
        return cpt

    def MPE(self, evidence: list, elimination_order) -> dict:
        """
        Returns the most probable explanation for the evidence

        Takes evidence as input and returns the most probable explanation for the evidence as an instantiation
        """
        evidence_vars = [var for (var, _) in evidence]

        # prune network
        pruned_network = self.pruner([], evidence_vars)

        # get al variables
        vars = pruned_network.get_all_variables()

        # get elimination order
        elimination_order = list(elimination_order(vars))

        # condition all CPTs
        cpts = [self.condition(cpt, evidence)
                for cpt in pruned_network.get_all_cpts().values()]

        for i in range(len(vars)):
            var = elimination_order[i]
            cpts_with_var = [cpt for cpt in cpts if var in cpt.columns]
            if len(cpts_with_var) > 0:
                product = cpts_with_var[0]
                for cpt in cpts_with_var[1:]:
                    product = self.multiply_cpts(product, cpt)

                # max over var
                if len(product.columns) > 2:
                    max = self.maxing_out(product, [var])
                else:
                    max = product

                # replace factors in cpts with max
                for i in range(len(cpts)):
                    for cpt in cpts_with_var:
                        if cpt.equals(cpts[i]):
                            cpts[i] = max

            result = cpts[0]
            for cpt in cpts[1:]:
                print(
                    f'------------------\n{result}\n\n{cpt}\n------------------')
                result = self.multiply_cpts(result, cpt)

        return result

    def MAP(self, M: list, evidence: list, elimination_order):
        """
        Returns the 'most a posteriori estimate' for the given variables and evidence
        """
        # prune network
        pruned_network = self.pruner(M, evidence)

        # get al variables
        vars = [var for var in pruned_network.get_all_variables()
                if var not in M]

        # get elimination order
        elimination_order = list(elimination_order(
            vars)) + list(elimination_order(M))

        # get all factors
        cpts = [self.condition(cpt, evidence)
                for cpt in pruned_network.get_all_cpts().values()]

        for i in range(len(vars)):
            # calc product over relevant factors
            var = elimination_order[i]
            cpts_with_var = [cpt for cpt in cpts if var in cpt.columns]
            for cpt in cpts_with_var:
                print(f'-------------------\n{cpt}\n-------------------')
            product = cpts_with_var[0]
            for cpt in cpts_with_var[1:]:
                print('.')
                product = self.multiply_cpts(product, cpt)

            # replace relevant factors with max or sum
            if var in M:
                factor = self.maxing_out(product)
            else:
                factor = self.summing_out(product)
            for i in range(len(cpts)):
                if cpts[i] in cpts_with_var:
                    cpts[i] = factor

        return cpts


# test maxing out
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
example_cpt = bn_grass.bn.get_cpt('Wet Grass?')
print(example_cpt)
bn_grass.maxing_out(example_cpt, ['Wet Grass?'])
'''

# test summing out
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
example_cpt = bn_grass.bn.get_cpt('Wet Grass?')
print(example_cpt)
bn_grass.summing_out(example_cpt, ['Wet Grass?'])
'''

# test pruner
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
bn_grass.bn.draw_structure()
pruned_bn_grass = bn_grass.pruner({'Winter?', 'Wet Grass?'},{'Sprinkler?'})
pruned_bn_grass.bn.draw_structure()
'''

# test JPD^maxing-out
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
print(bn_grass.maxing_out(('Sprinkler?', 'Rain?')))
'''

# test JPD^summing-out
'''
bn_grass = BNReasoner('testing/lecture_example.BIFXML')
print(bn_grass.summing_out(('Slippery Road?', 'Sprinkler?', 'Rain?')))
'''

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
if __name__ == "__main__":
    # test pruner
    '''
    bn_grass = BNReasoner('testing/lecture_example.BIFXML')
    bn_grass.bn.draw_structure()
    pruned_bn_grass = bn_grass.pruner({'Winter?', 'Wet Grass?'},{'Sprinkler?'})
    pruned_bn_grass.bn.draw_structure()
    '''

    # test maxing-out
    '''
    bn_grass = BNReasoner('testing/lecture_example.BIFXML')
    print(bn_grass.maxing_out(('Sprinkler?', 'Rain?')))
    '''

    # test summing-out
    '''
    bn_grass = BNReasoner('testing/lecture_example.BIFXML')
    print(bn_grass.summing_out(('Slippery Road?', 'Sprinkler?', 'Rain?')))
    '''

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

    # test d-separation
    '''
    reasoner = BNReasoner('testing/dog_problem.BIFXML')
    BN = BayesNet()
    network = BN.load_from_bifxml('testing/dog_problem.BIFXML')
    BN.draw_structure()
    test = reasoner.d_separation(
        network, 'family-out', 'hear-bark', ['dog-out'])
    print(test)

    # test = reasoner.d_separation(network, 'dog-out', 'light-on', ['dog-out'])
    # print(test)
    '''

    # test MPE
    """
    bn = BayesNet()
    bn.load_from_bifxml('testing/lecture_example.BIFXML')
    bnr = BNReasoner(bn)
    bnr.MPE([('Winter?', True), ('Wet Grass?', True)], bnr.random_ordening)
    """
