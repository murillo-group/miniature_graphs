#!/usr/bin/env python3
from minigraphs import Metropolis
from numpy import log
import pandas as pd 
import networkx as nx
import sys
import os
import json
'''Calculates the weights for the specified graph
'''

# Validate inputs
try:
    graph_name = sys.argv[1]
    frac_size = round(float(sys.argv[2]),3)
    n_iterations = int(sys.argv[3])
    N = int(sys.argv[4])
    
except IndexError:
    print("Error: not enough input arguments provided. Necessary inputs are\n")
    print("\t - Graph name\n\t - frac_size\n\t - n_iterations\n\t - n_samples")

# Retrieve Graph Metrics
DATA_DIR = os.environ['DATA_DIR']
NET_DIR = os.path.join(DATA_DIR,'networks',graph_name)
PARAMS_DIR = os.path.join(NET_DIR,'parameters')

input_file = os.path.join(NET_DIR,'metrics.json')
with open(input_file) as file:
    metrics = json.load(file)
    metrics_target = {key:metrics[key] for key in ['density','assortativity_norm','clustering']}

# Specify Metric funcions
funcs_metrics = {
    'density': nx.density,
    'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
    'clustering': nx.average_clustering
}

# Calculate miniature size
try:
    n_vertices = int(metrics['n_vertices'] * frac_size)
    
    if (n_vertices < 1):
        raise ValueError
    
except ValueError:
    print("Error: Invalid number of vertices in the miniature")
    
print(f"Calculating Miniaturization Parameters for {graph_name}")
print(f"\t - Size: {n_vertices} nodes ({frac_size * 100}% miniaturization)")
print(f"\t - Number iterations per sample: {n_iterations}")
print(f"\t - Number of samples: {N}\n")

# Calculate weights
params = []
for i in range(N):
    print(f"Sweep {i+1}/{N}")
    replica = Metropolis(0,
                         funcs_metrics,
                         n_iterations=n_iterations,
                         n_changes=10
                        )
    
    G = nx.erdos_renyi_graph(n_vertices,metrics_target['density'])
    replica.transform(G,metrics_target)

    df = replica.trajectories_.copy()
    weights = dict(1/df[replica.metrics].diff().abs().mean())

    print("Weights:")
    print(json.dumps(weights,indent=4))

    # Calculate optimal beta
    replica = Metropolis(0,
                        funcs_metrics,
                        n_iterations=n_iterations,
                        metrics_weights=weights,
                        n_changes=10
                        )
    
    G = nx.erdos_renyi_graph(n_vertices,metrics_target['density'])
    replica.transform(G,metrics_target)

    df = replica.trajectories_.copy()

    beta = -log(0.23) * 1/df['Energy'].diff().abs().mean()

    print(f"Beta: {beta}\n")

    params.append([beta] + list(weights.values()))

params = pd.DataFrame(params,columns=['beta'] + list(weights.keys()))
print("Measeured parameters:")
print(params,f"\n")

params = params.mean()
print("Final parameters:")
print(params)

# Create outpot directory
if not os.path.exists(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)

# Save to JSON
file_name = os.path.join(PARAMS_DIR,f'params_{frac_size}.json')
with open(file_name,'w+') as f:
    json.dump(dict(params),f,indent=4)