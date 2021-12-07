from BayesNet import BayesNet
import pandas as pd
import copy

BN = BayesNet()
BN.load_from_bifxml('testing/dog_problem.BIFXML')


def multiply_cpts(cpt_1, cpt_2):
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


cpt_1 = BN.get_cpt("light-on")
cpt_2 = BN.get_cpt("dog-out")
print(cpt_1)
print(cpt_2)
t = multiply_cpts(cpt_1, cpt_2)
print(t)
