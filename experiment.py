'''Code for running the experiments'''
from BNReasoner import BNReasoner
import pandas as pd
size_growth_iterations = 100

BR = BNReasoner("examples/dog_problem")
heuristics = ['random', 'min-edge', 'min-fill']

result = pd.DataFrame(columns=["network size"] + heuristics)

network = None
for iteration in range(size_growth_iterations):
    network = grow_network(network)  # Grow previous network
    for heuristic in heuristics:
        # Set the network of our reasoner to be the grown network
        BR.bn = network
