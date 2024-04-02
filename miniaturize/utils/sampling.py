"""
Random Graph Sampling functions for Miniaturize
"""

import networkx as nx
import pandas as pd
import numpy as np

def sample_generator(generator,params,metrics,num_samples=10):
    """
    Samples the specified NetworkX generator 
    """

    def get_metrics(G,metrics):
        """Return a list of network metrics of the specified graph"""
        return [metric(G) for metric in metrics]

    # Determine the column names
    columns = ["Generator"] + [metric.__name__ for metric in metrics] + ["parameters"]

    # Sample the generator
    measurements = []
    for i in range(0,num_samples):
        # Store data
        data = [generator.__name__] + get_metrics(generator(*params),metrics) + [params]
        # Instantiate graph
        measurements.append(data)

    return pd.DataFrame(measurements, columns=columns)

def grid_sample(generator,params_grid,metrics,num_samples=10):
    """
    Samples the specified NetworkX generator across a wide range of parameters
    """

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Get dimensions of each parameter array
    shape = [len(param) for param in params_grid]

    # Index array
    idx = np.argwhere(np.ones(shape,dtype=bool))

    # Sample array
    for idx_local in idx:
        # Select sampling parameters
        params = [params_grid[j][flag] for j,flag in enumerate(idx_local)]

        # Sample generator
        sample = sample_generator(generator,params,metrics,num_samples)

        # Append to existing dataframe
        df = pd.concat([df,sample],ignore_index=True)

    return df