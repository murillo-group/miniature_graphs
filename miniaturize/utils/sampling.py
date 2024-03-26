import networkx as nx
import pandas as pd
import numpy as np

def get_metrics(G):
    C = nx.average_clustering(G)
    r = nx.assortativity.degree_assortativity_coefficient(G)

    return [C,r]

def get_name(generator):
    return generator.__name__

def sample_generator(generator,params,n=50):
    """
    dict:           Graph generator to be sampled
    params:         Parameters for the generator
    n:              Number of samples to be drawn from each parameter combination
    """

    metrics = np.zeros((n,2))
    for i in range(0,n):
        metrics[i,:] = get_metrics(generator(*params))

    data = {'name':[get_name(generator)] * n,
            'm1':metrics[:,0],
            'm2':metrics[:,1]}

    return pd.DataFrame(data)

def grid_sample(generator,params_grid,num_samples=10):
    """
    Samples the specified generator in a grid-like manner
    generator:      Handle to the generator to sample from
    params:         parameters to sample from the generator
    """

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Get number of parameters
    num_parameters = len(params_grid)

    # Get dimensions of each parameter array
    shape = [len(param) for param in params_grid]

    # Index array
    idx = np.argwhere(np.ones(shape,dtype=bool))

    # Sample array
    for idx_local in idx:
        # Select sampling parameters
        params = [params_grid[j][flag] for j,flag in enumerate(idx_local)]

        # Sample generator
        sample = sample_generator(generator,params,num_samples)

        # Append parameters
        sample['graph_size'] = params[0]
        sample['parameters'] = [params[1:]] * (num_samples)

        # Append to existing dataframe
        df = pd.concat([df,sample],)

    return df.reset_index()