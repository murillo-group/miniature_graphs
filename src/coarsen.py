#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
from minigraphs.Metropolis import CoarseNET
from scipy.sparse import load_npz, save_npz
import os
import sys
import json
import yaml
import networkx as nx
import datetime

def load_params(file):
    '''Load parameters from parameter file
    '''
    with open(file,'r') as file:
        return yaml.safe_load(file)
    
@click.command()
@click.option('--param-file',type=click.Path(exists=True),help="Path to configuration file")
@click.option('--graph-file',type=click.Path(exists=True),help="Path to .npz file containing graph")
@click.option('--alpha',type=float,help="Reduction parameter")
@click.option('--out-dir',type=click.Path(),default=".",help="Output directory")
def coarsen(param_file,
            graph_file,
            alpha,
            out_dir):
    
    # Load parameters from file
    config = load_param(param_file) if param_file else {}
    
    # Use parameter file values unless overriden by CLI
    params = {
        'graph_file': graph_file or config.get('graph_file',{}),
        'alpha': alpha or config.get('alpha',{}),
        'out_dir': out_dir or config.get('out_dir',{})
    }
    
    # Validate parameters
    missing_params = [key for key, value in params.items() if value is None]
    if missing_params:
        raise click.UsageError(f"Missing required parameters: {','.join(missing_params)}")
    
    if not (0 < alpha < 1):
        raise click.BadParameter('Shrinking parameter must be in range (0,1)')
    
    # Load Graph
    G = nx.from_scipy_sparse_array(load_npz(graph_file))
    
    # Coarsen Graph
    coarsener = CoarseNET(alpha,G)
    coarsener.coarsen()
    
    # Save graph in output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    save_npz('coarse.npz',nx.to_scipy_sparse_array(coarsener.G_coarse_))
    
if __name__ == '__main__':
    coarsen()


