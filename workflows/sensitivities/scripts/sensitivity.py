#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import networkx as nx
from minigraphs import simulation as sim 
from minigraphs.utils import dispatch
import numpy as np
import matplotlib.pyplot as plt
import click
import pandas as pd
import yaml
from inspect import signature

# Load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
def get_function_arguments(func):
    # Get the signature of the function
    signature = inspect.signature(func)
    # Extract the parameter names from the signature
    return [param.name for param in signature.parameters.values()]

@click.command()
@click.option('--config-file',type=click.Path(exists=True))
@click.option('--output',type=click.Path(exists=True),defualt='.',help="Output file")
def call(config_file,
         output)


def sensitivity_analysis(parameter,
                         n_graphs,
                         n_trials,
                         n_vertices,
                         n_steps):
    # Validate input
    config = {}
    if config_file:
        config = load_config(config_file)
        
    # Check all arguments are provided
    for argument in arguments():
        if 
        
    # Dictionary of generators
    gen_dict = {
        "density": (lambda n,p: nx.erdos_renyi_graph(n,p), 10 ** np.linspace(-3,-1,10)),
        "clustering": (lambda n,p: nx.erdos_renyi_graph(n,p), np.arange(0.05,0.5,0.025))
    }
    
    # Select generator according to parameter
    generator, datapoints = gen_dict[parameter]
    
    # Dictionary of qois
    qois_dict = sim.qois_sir

    # Sir simulation
    sir = sim.Sir(0.1,0.1)

    # Sensitivity object
    # Allocate memory for data
    n_datapoints = datapoints.shape[0]
    n_qois = len(list(qois_dict.keys()))
    data = np.zeros((n_datapoints,n_graphs,n_trials,n_qois))
    
    for k, p in enumerate(datapoints):
        for i in range(n_graphs):
            # Generate graph
            G = generator(n_vertices,p)
            A = nx.to_scipy_sparse_array(G)
            
            # Instantiate simulation object
            simulation = sim.Simulation(A)
            for j in range(n_trials):
                # Run simulation
                simulation.run(sir, n_steps)
                
                # Calculate qois
                data[k,i,j,:] = np.array(list(dispatch(simulation.trajectories_df_,qois_dict).values()))
                
    # Calculate mean and standard deviation
    mean = pd.DataFrame(data.mean(axis=(1,2)),columns=list(qois_dict.keys()))
    sigma = pd.DataFrame(data.std(axis=(1,2)),columns=list(qois_dict.keys()))
    
    mean[parameter] = datapoints
    sigma[parameter] = datapoints
    
    # Save data
    mean.to_csv(f"{parameter}_mean.csv",sep=',',index=False)
    mean.to_csv(f"{parameter}_std.csv",sep=',',index=False)
    
    
if __name__ == "__main__":
    sensitivity_analysis()