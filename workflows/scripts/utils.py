from scipy.sparse import load_npz, save_npz
import networkx as nx
from yaml import safe_load, dump

def save_graph(file,G):
    '''Saves the adjacency matrix of a graph'''
    save_npz(file,nx.to_scipy_sparse_array(G))

def load_graph(file):
    '''Loads the adjacency matrix of a graph'''
    return nx.from_scipy_sparse_array(load_npz(file))

def save_dict(file,dictionary):
    '''Saves a dictionary in yaml format
    '''
    with open(file,'w') as file: 
        dump(dictionary,file,default_flow_style=False)
        
def load_dict(file):
    '''Loads a dictionary from a file
    '''
    with open(file,'r') as file:
        return safe_load(file)
    
    