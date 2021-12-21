from typing import Dict, List, Tuple, Union
from BayesNet import BayesNet
from copy import copy, deepcopy

import logging
import networkx as nx
import pandas as pd
import random
import numpy as np


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet], log_level=logging.CRITICAL):
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

        # init logger
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=log_level)

    def d_separation(self, network, x, y, z):
        z_parents = self.bn.get_parents(z)
        z_children = self.bn.get_children(z)
        # print(z_children)
        # print(z_parents)
        nodes_to_visit = [(x, 'asc')]
        already_visited = set()
        nodes = set(self.bn.get_all_variables())

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

    def order_min_degree(self, network: BayesNet) -> List[str]:
        """
        Orders nodes by their degree (smallest first)

        Returns a list of nodes (str)
        """
        degrees = network.get_interaction_graph().degree()
        degrees = sorted(degrees, key=lambda x: x[1])
        order = [x[0] for x in degrees]
        return order

    def order_min_fill(self, network: BayesNet) -> List[str]:
        """
        Orders nodes such that elimination leads to the fewest new edges

        Returns alist of nodes (str)
        """
        int_graph = network.get_interaction_graph()
        new_edges = []
        for node in int_graph:
            n = 0
            neighbors = int_graph.neighbors(node)
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 == n2:
                        continue
                    if n2 not in int_graph.neighbors(n1):
                        n += 1
            new_edges.append((node, n))
        new_edges = sorted(new_edges, key=lambda x: x[1])
        return [x[0] for x in new_edges]

    def order_random(self, network: BayesNet) -> List[str]:
        """
        Returns a random order of the nodes
        """
        vars = network.get_all_variables()
        random.shuffle(vars)
        return vars

    def prune(self, query: List[str], evidence: Dict[str, bool]) -> BayesNet:
        """
        """
        new_bn = deepcopy(self.bn)

        # loop until no changes are made
        changes = True
        while changes:
            changes = False
            # prune leaf nodes that are not in query and evidence
            for var in new_bn.get_all_variables():
                if new_bn.get_children(var) == [] and var not in query and var not in evidence:
                    new_bn.del_var(var)
                    changes = True

            cpts = new_bn.get_all_cpts()
            for evidence_var, assignment in evidence.items():
                # update cpts
                for variable in new_bn.get_all_variables():
                    cpt = cpts[variable]
                    if evidence_var not in cpt.columns:
                        continue
                    drop_indices = cpt[cpt[evidence_var] != assignment].index
                    new_cpt = cpt.drop(drop_indices)
                    new_bn.update_cpt(variable, new_cpt)
                # remove outgoing edges from nodes in evidence
                for child in new_bn.get_children(evidence_var):
                    changes = True
                    new_bn.del_edge((evidence_var, child))

        return new_bn

    def is_unique(self, s):
        '''Quick check if all values in df are equal'''
        if not isinstance(s, pd.DataFrame):
            s.to_frame()  # If we got a series object transform it to DF
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    def multiply_cpts_extensive(self, cpt_1, cpt_2):
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
            p_1 = deepcopy(cpt_1)
            p_2 = deepcopy(cpt_2)

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
                current_table = self.multiply_cpts_extensive(
                    current_table, self.bn.get_cpt(ancestor))
            results.append(current_table)

        # Then multiply those two final resulting vars in Q
        end = results[0]
        for j in range(1, len(results)):
            end = self.multiply_cpts_extensive(end, results[j])

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

    def summing_out(self, cpt: pd.DataFrame, sum_out_variables: List[str], assignment: Dict[str, bool]) -> pd.DataFrame:
        """
        Takes set of variables (given als list of strings) that needs to be
        summed out as an input and returns table with without given variables
        when applied to a Bayesian Network
        """
        # delete columns of variables that need to be summed out
        dropped_cpt = cpt.drop(columns=sum_out_variables)

        # get the variables still present in the table
        remaining_variables = list(dropped_cpt.columns.values)[:-1]

        # return trivial factor if no variables left
        if len(remaining_variables) == 0:
            return cpt['p'].sum()

        # sum up p values if rows are similar
        PD_new = dropped_cpt.groupby(
            remaining_variables).aggregate({'p': 'sum'})
        PD_new.reset_index(inplace=True)

        return PD_new

    def maxing_out(self, cpt: pd.DataFrame, max_out_variables: List[str], assignment: Dict[str, bool]) -> pd.DataFrame:
        """
        Takes set of variables (given als list of strings) that needs to be
        maxed out as an input and returns table with without given variables
        when applied to a Bayesian Network
        """
        # delete columns of variables that need to be maxed out
        dropped_cpt = cpt.drop(columns=max_out_variables)

        # get the variables still present in the table
        remaining_variables = list(dropped_cpt.columns.values)[:-1]

        # return assignment if no variables left
        if len(remaining_variables) == 0:
            max_id = cpt['p'].idxmax()
            for var in cpt.columns.values[:-1]:
                assignment[var] = cpt[var][max_id]
            return None

        # take max p value for remaining rows if they are similar
        PD_new = dropped_cpt.groupby(
            remaining_variables).aggregate({'p': 'max'})
        PD_new.reset_index(inplace=True)
        return PD_new

    def multiply_factors(self, cpt1: pd.DataFrame, cpt2: pd.DataFrame) -> pd.DataFrame:
        """
        """
        # Make sure cpt1 has the most columns
        if len(cpt2.columns) > len(cpt1.columns):
            cpt1, cpt2 = cpt2, cpt1

        # Multiply the two CPTs
        for var in cpt2.columns[:-1]:
            if var not in cpt1.columns:
                continue
            for _, row2 in cpt2.iterrows():
                t_value = row2[var]
                for i, row1 in cpt1.iterrows():
                    if row1[var] == t_value:
                        cpt1.at[i, 'p'] *= row2['p']

                #indices = cpt1[var] == t_value
                #cpt1.loc[cpt1[var] == t_value, 'p'] *= row['p']

        return cpt1

    def multiply_n_factors(self, cpts: List[pd.DataFrame]) -> pd.DataFrame:
        """
        """
        if len(cpts) > 1:
            result = cpts[0]
            for cpt in cpts[1:]:
                result = self.multiply_factors(result, cpt)
        else:
            result = cpts[0]
        return result

    def condition(self, cpt: pd.DataFrame, evidence: Dict[str, bool]) -> pd.DataFrame:
        """
        Given a CPT and evidence, returns a conditioned CPT
        """
        for (var, value) in evidence.items():
            if var in cpt.columns:
                cpt = cpt.loc[cpt[var] == value]
        return cpt

    def MPE(self, evidence: Dict[str, bool], order_function=order_random):
        """
        """
        logging.info('Starting MPE')
        logging.info('Starting pruning')
        pruned_network = self.prune([], evidence)
        assignment = dict()

        # get elimination order
        logging.info('Getting elimination order')
        if order_function in [self.order_random, self.order_min_degree, self.order_min_fill]:
            elimination_order = order_function(pruned_network)
        else:
            return "Error: order_function not recognized"

        # get and condition cpts
        logging.info('Getting and conditioning CPTs')
        cpts = dict()
        for var, cpt in pruned_network.get_all_cpts().items():
            cpts[var] = self.condition(cpt, evidence)

        logging.info('Starting inference')
        for var in elimination_order:
            logging.info(f'Inference on var: {var}')
            # get all cpts in which var occurs
            logging.info(f'    Getting relevant CPTs')
            fks = [key for key, cpt in cpts.items() if var in cpt.columns]
            fks_cpt = [cpts[key] for key in fks]

            if len(fks) == 0:
                continue
            # calc product of cpts
            logging.info(f'    Multiplying CPTs')
            f = self.multiply_n_factors(fks_cpt)

            # max out f
            logging.info(f'    Maxing out CPT')
            fi = self.maxing_out(f, [var], assignment)

            # replace cpts
            for key in fks:
                cpts.pop(key)
            if fi is not None:
                cpts['+'.join(fks)] = fi

        return assignment

    def MAP(self, query: List[str], evidence: Dict[str, bool], order_function=order_random):
        """
        """
        pruned_network = self.prune(query, evidence)
        assignment = dict()

        # get elimination order
        if order_function in [self.order_random, self.order_min_degree, self.order_min_fill]:
            temp_elimination_order = order_function(pruned_network)
            elimination_order_1 = [
                x for x in temp_elimination_order if x not in query]
            elimination_order_2 = [
                x for x in temp_elimination_order if x in query]
            elimination_order = elimination_order_1 + elimination_order_2
        else:
            return "Error: order_function not recognized"

        # get and condition cpts
        cpts = dict()
        for var, cpt in pruned_network.get_all_cpts().items():
            cpts[var] = self.condition(cpt, evidence)

        for var in elimination_order:
            # get all cpts in which var occurs
            fks = [key for key, cpt in cpts.items() if var in cpt.columns]
            fks_cpt = [cpts[key] for key in fks]

            if len(fks) == 0:
                continue
            # calc product of cpts
            f = self.multiply_n_factors(fks_cpt)

            if var in query:
                # max out f
                fi = self.maxing_out(f, [var], assignment)
            else:
                # sum out f
                fi = self.summing_out(f, [var], assignment)

            # replace cpts
            for key in fks:
                cpts.pop(key)
            if fi is not None:
                cpts['+'.join(fks)] = fi

        return assignment
