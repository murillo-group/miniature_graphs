import minigraphs
import pandas as pd
import networkx as nx
from importlib import reload

# Define functions
assortativity = lambda G: nx.degree_assortativity_coefficient(G)/2

metrics_funcs = {
    'density': nx.density,
    'assortativity': assortativity,
    'clustering': nx.average_clustering
}

# Instantiate replica
replica = minigraphs.Metropolis(0,
                     metrics_funcs,
                     50)

# Transform replica
metrics_target = {
    "density": 5.20e-04,
    "clustering": 5.20e-02,
    "assortativity": -0.15
}

G = nx.erdos_renyi_graph(300,0.1)

replica.transform(G,metrics_target,verbose=True)
print(replica)
df = replica.trajectories_

print(df)