import click
from functions import load_config, get_arguments, sample, mkdir
from os.path import join
from networkx import to_scipy_sparse_array
from scipy.sparse import save_npz

@click.command()
@click.argument('output',type=click.Path())
@click.argument('config-file',type=click.Path(exists=True))
def call(output,
         config_file):
    # Load script configuration
    config = {}
    if config_file:
        config = load_config(config_file)

    # Get function arguments
    arguments = get_arguments(sample)
    
    # Verify arguments
    for argument in arguments:
        if config.get(argument,None) is None:
            message = f"Input argument '{argument}' missing from parameter file"
            raise ValueError(message)
        
    # Evaluate function
    data, metadata = sample(**config)
    
    # Create output directory
    mkdir(output)
    
    # Write data
    file_names = [""] * metadata.shape[0]
    for i, graph in enumerate(data):
        file_names[i] = f"graph_{i:02d}.npz"
        save_npz(join(output,file_names[i]),to_scipy_sparse_array(graph))
    
    # Write metadata
    metadata['file_name'] = file_names
    metadata.set_index('file_name',inplace=True)
    metadata.to_csv(join(output,'metadata.csv'))
    
if __name__ == "__main__":
    call()