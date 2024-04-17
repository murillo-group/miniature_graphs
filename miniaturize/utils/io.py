"""
Input and output functions for miniaturize
"""

import os
import networkx as nx
from scipy import io
from glob import glob

def load_graph(path):
    # Write path to graph
    pattern = os.path.join(path,"graph*")

    # Find file 
    filename = glob(pattern)[0]
    if len(filename) == 0:
        print("Error: file not found")

    # Get extension
    ext = os.path.splitext(filename)[1]

    # Load graph
    path_full = os.path.join(path,filename)
    print("Loading " + path_full)
    if ext == ".edgelist":
        print("Loading graph from edgelist")
        G = nx.read_edgelist(path_full,nodetype=int,delimiter=',')
    elif ext == ".mtx":
        print("Loading graph from matrix market file")
        G = nx.Graph(io.mmread(path_full))
    else:
        raise Exception(f"Invalid graph format {ext}")

    # Print message
    print("Loaded graph " + path)

    return G

