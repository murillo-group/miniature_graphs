import numpy as np
import pandas as pd

# Load simulation runs
results = np.load(snakemake.input[0])

qois_arrays = {
    'time-to-peak': results[:,1,:].argmax(1,keepdims=True),
    'peak': results[:,1,:].max(1,keepdims=True),
    'size': results[:,2,-1]
}

qois = []
for quantity, array in qois_arrays.items():
    temp = {
        'mean': array.mean(0),
        'std': array.std(0),
        'min': array.min(0),
        'max': array.max(0),
    }
    
    # Store qois
    qois.append(temp)
    
# Construct dataframe
df = pd.DataFrame(qois,index=qois_arrays.keys())

# Save dataframe to csv file
df.to_csv(snakemake.output[0])