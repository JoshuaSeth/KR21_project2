import pandas as pd 
from scipy.stats import f_oneway

results = pd.read_csv('results.csv')
results = results[results.nodes != 80] # delete rows where nodes = value 80, ran for a little, but ended up takin too long
mpe_results = results[results.algorithm == 'MPE']
map_results = results[results.algorithm == 'MAP']
print(mpe_results)

# selecting relevant data from dataframe 
random_10 = results[(results["order"] == "order_random") & (results["nodes"] == 10)]['time']
min_degree_10 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 10)]['time']
min_fill_10 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 10)]['time']

random_20 = results[(results["order"] == "order_random") & (results["nodes"] == 20)]['time']
min_degree_20 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 20)]['time']
min_fill_20 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 20)]['time']

random_30 = results[(results["order"] == "order_random") & (results["nodes"] == 30)]['time']
min_degree_30 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 30)]['time']
min_fill_30 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 30)]['time']

random_40 = results[(results["order"] == "order_random") & (results["nodes"] == 40)]['time']
min_degree_40 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 40)]['time']
min_fill_40 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 40)]['time']

random_50 = results[(results["order"] == "order_random") & (results["nodes"] == 50)]['time']
min_degree_50 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 50)]['time']
min_fill_50 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 50)]['time']

random_60 = results[(results["order"] == "order_random") & (results["nodes"] == 60)]['time']
min_degree_60 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 60)]['time']
min_fill_60 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 60)]['time']

random_70 = results[(results["order"] == "order_random") & (results["nodes"] == 70)]['time']
min_degree_70 = results[(results["order"] == "order_min_degree") & (results["nodes"] == 70)]['time']
min_fill_70 = results[(results["order"] == "order_min_fill") & (results["nodes"] == 70)]['time']

print(min_fill_70)
print(min_degree_70)

#x, y = random_60, min_degree_60
#x, y = random_60, min_fill_60
x, y = min_degree_60, min_fill_60

F, p = f_oneway(x, y)
if p < 0.05:
    print('significant, p:', p)
else:
    print('no, p:', p)

'''results: significant are ...'''
# 30 only random_30, min_degree_30
# 40 not min_degree_40, min_fill_40
# 50 not min_degree_50, min_fill_50
# 60 not min_degree_60, min_fill_60
# 70 not min_degree_70, min_fill_70