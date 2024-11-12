from minigraphs import simulation as sim
from scipy.sparse import load_npz
import numpy as np

# Instantiate sir object
sir = sim.Sir(snakemake.params.tau, snakemake.params.gamma)

# Instantiate simulation object
simulation = sim.Simulation(load_npz(snakemake.input[0]))

# Allocate memory
shape = (snakemake.params.n_trials, 3, snakemake.params.n_steps)
results = np.zeros(shape)

# Simulate epidemic
for i in range(snakemake.params.n_trials):
    simulation.run(sir,snakemake.params.n_steps)
    
    results[i,:,:] = simulation.trajectories_.T
    
# Save simulation results
np.save(snakemake.output[0],results)