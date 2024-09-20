#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd 
import numpy as np

DATA_DIR = os.path.join(os.getenv('DATA_DIR'),'databases')

'''
files = glob.iglob(".txt",root_dir=DATA_DIR)
for file in files:
    df = pandas
'''

DIR = os.path.join(DATA_DIR,'social')

def read_dbs(path):
    def checkfiles(sublist,list):
        return set(sublist) <= set(list)
    
    files = glob.glob("*csv",root_dir=path)
    if checkfiles(['names.csv','values.csv'],files) is True:
        # Read data frames
        df1 = pd.read_csv(os.path.join(path,'names.csv'))
        df2 = pd.read_csv(os.path.join(path,'values.csv'),delimiter="\t")
        
        # Construct new labels
        labels = ['num_vertices',
                  'num_edges',
                  'degree_max',
                  'degree_avg',
                  'assortativity_coeff',
                  'num_triangles',
                  'avg_num_triangles',
                  'max_triangles_per_edge',
                  'clustering_local_avg',
                  'clustering_global',
                  'max_k_core',
                  'low_max_clique',
                  'size']

        # Rename columns of values df
        df2.drop(columns=df2.columns[-2:],inplace=True)
        df2.rename(columns=dict(zip(df2.columns,labels)),inplace=True)

        # Join data frames
        df = df1.join(df2)

    else:
        raise KeyError("Files not in directory")

    return df

def format_dbs(df):
    def to_numeric(series):
        # Parse strings
        df = series.str.split(r'(?=[A-Za-z])',expand=True)

        # Scale according to suffixes
        if df.shape[1] > 1:
            mapping = {'K':1e3,
                       'M':1e6,
                       None:1}

            s = df.iloc[:,1].map(mapping)
        
        else:
            s = 1

        return pd.to_numeric(df.iloc[:,0],errors='coerce') * s

    for column in df.columns[1:]:
        df[column] = to_numeric(df[column])

df = read_dbs(DIR)
format_dbs(df)
df.to_csv(os.path.join(DIR,"df.csv"),sep=",")

print(df)


    
