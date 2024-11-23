from utils import load_dict
import pandas as pd

# INPUTS
files = snakemake.input.files
df_file = snakemake.output[0]

# AGGREGATE DICTIONARIES
parameters = [load_dict(file) for file in files]
parameters = pd.DataFrame(parameters)
parameters.rename(lambda idx: f"graph_{idx}",inplace=True)

# SAVE DF
parameters.to_csv(df_file)