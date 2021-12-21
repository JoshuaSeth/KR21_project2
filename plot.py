import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    results = pd.read_csv('results.csv')
    results = results[results.Nodes != 80] # delete rows where nodes = value 80, ran for a little, but ended up takin too long
    mpe_results = results[results.algorithm == 'MPE']
    map_results = results[results.algorithm == 'MAP']

    sns.lineplot(data=mpe_results, x='Nodes', y = 'time (s)', hue='order')
    plt.show()
    sns.lineplot(data=map_results, x='Nodes', y = 'time (s)', hue='order')
    plt.show()