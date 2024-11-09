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
import os

# Load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
def get_arguments(func):
    # Extract the parameter names from the signature
    return [param.name for param in signature(func).parameters.values()]

def mkdir(output_dir):
    '''Creates output directory if it doesn't exist already
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def sample(parameter,
           inputs,
           n_graphs,
           n_vertices):
    '''Generates a an ensamble of graphs according to the indicated distribution
    '''
    def get_error(metric,target):
        '''Calculates the error in a generated metric
        '''
        return np.abs(metric - target) / target * 100
    
    inputs = eval(inputs)
    
    # Dictionary of generators
    generators = {
        "density": lambda n,p: nx.erdos_renyi_graph(n,p),
        "clustering": lambda n,p: nx.erdos_renyi_graph(n,p)
    }
    
    metrics_dict = {
        "density": lambda G: nx.density(G),
        "clustering": lambda G: nx.average_clustering(G),
        "assortativity": lambda G: nx.degree_assortativity_coefficient(G)
    }
    
    # Select generator according to parameter
    generator = generators[parameter]

    # Allocate memory for data
    n_inputs = inputs.shape[0]
    graphs = [0] * n_inputs
    columns = list(metrics_dict.keys()) + ["error"]
    n_columns = len(columns)
    
    graphs = np.empty((n_inputs,n_graphs),dtype=object)
    parameters = np.zeros((n_inputs,n_graphs,n_columns))
    
    for i, p in enumerate(inputs):
        for j in range(n_graphs):
            # Generate graph
            graphs[i,j] = generator(n_vertices,p)
            
            # Calculate metrics
            metrics = {key: func(graphs[i,j]) for key, func in metrics_dict.items()}
            error = get_error(metrics[parameter],p)
            parameters[i,j,:] = list(metrics.values()) + [error]
    
    # Obtain data and metadata
    graphs = graphs.reshape(-1,1).squeeze()
    parameters = parameters.reshape(-1,n_columns).squeeze()
    parameters = pd.DataFrame(parameters,columns=columns)
    parameters["parameter"] = parameter
    
    return graphs, parameters

def index

def characterize_graph(graphs,characteristics):
    '''Characterizes a set of graphs
    '''
    pass
    
def simulate_sir(graph,
                 tau,
                 gamma,
                 n_iterations):
    '''Simulates the sir model on the specified graphs
    '''
    pass




