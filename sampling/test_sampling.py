#!/usr/bin/env python3
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import pandas as pd 

import sys
sys.path.append("../../../")

import miniaturize.utils.sampling as sampling

def network_size(G):
    return len(G.nodes)

# Define a handle to the metrics to find
list_metrics = [nx.average_clustering,
                nx.degree_assortativity_coefficient,
                nx.density,
                network_size]

# Select generator
gen = nx.erdos_renyi_graph

# Define generator parameters
params = (100,0.5)

# Sample generator
print(sampling.sample_generator(gen,params,list_metrics))

# Define grid parameters
params_grid = [(100,200),[0.5]]
print(sampling.grid_sample(gen,params_grid,list_metrics))



