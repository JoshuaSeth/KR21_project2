'''Code for running the experiments'''
from BNReasoner import BNReasoner
import pandas as pd
import time
import itertools

size_growth_iterations = 100

BR = BNReasoner("testing/dog_problem.bifxml")
heuristics = ['random', 'min-edge', 'min-fill']
queries = ['map', 'mpe']

# Get all possible combinations of query type and heuristic
combinations = list(itertools.product(heuristics, queries))
combinations = [a + " " + b for a, b in combinations]

# Set up results df
result = pd.DataFrame(columns=["network size"] + combinations)

network = None
# Grow network for number of iterations and perform tests on the same network
for iteration in range(1, size_growth_iterations+1):
    network = grow_network(network)  # Grow previous network
    result['network size'][iteration] = iteration*10
    for heuristic in heuristics:
        for query in queries:
            # Set the network of our reasoner to be the grown network
            BR.bn = network
            start_time = time.time()
            # Call the actual MAP/MPE
            if query == 'map':
                BR.map()
            else:
                BR.mpe()
            time = time.time() - start_time  # How much time did it take
            result[heuristic + query][iteration] = time
