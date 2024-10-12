#!/usr/bin/env python3
import click
from minigraphs import Metropolis
from numpy import log
import pandas as pd 
import networkx as nx
import sys
import os
import json
'''Calculates the parameters for the specified graph
'''

@click.command()
@click.argument('metrics_file',type=click.Path(exists=True))
@click.argument('frac_size',type=click.FLOAT)
@click.option('--output_dir', default="", help='Output directory.')
@click.option('--n_changes', default=10, help='Number of changes proposed at each iteration.')
@click.option('--n_samples', default=10, help='Number of times parameters are calculated.')
@click.option('--n_iterations', default=100, help='Number of iterations in each sample.')
def params(metrics_file,
           frac_size,
           output_dir,
           n_changes,
           n_samples,
           n_iterations):
    
    # Retrieve Graph Metrics
    with open(metrics_file) as file:
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
    
    print(f"Calculating parameters for graph at graph at {metrics_file}")
    print(f"\t - Size: {n_vertices} nodes ({frac_size * 100}% miniaturization)")
    print(f"\t - Number iterations per sample: {n_iterations}")
    print(f"\t - Number of samples: {n_samples}\n")

    # Calculate weights
    params = []
    for i in range(n_samples):
        print(f"Sweep {i+1}/{n_samples}")
        replica = Metropolis(0,
                            funcs_metrics,
                            n_iterations=n_iterations,
                            n_changes=n_changes
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
                            n_changes=n_changes
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

    # Save to JSON
    file_name = os.path.join(output_dir,f'params_{frac_size}.json')
    with open(file_name,'w+') as f:
        json.dump(dict(params),f,indent=4)
        
if __name__ == '__main__':
    params()