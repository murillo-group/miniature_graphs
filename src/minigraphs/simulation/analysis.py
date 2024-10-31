import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def plot_sir(df,*args,**kwargs):
    '''Plots the sir curves
    '''
    plt.plot('S',data=df)
    plt.plot('I',data=df)
    plt.plot('R',data=df)
    
def calculate_qois(df,dic):
    return {keys: func(df) for keys, func in dic.items()}
        
qois_sir = {
    "time-to-peak" : lambda df: df['I'].idxmax(),
    "peak" : lambda df: df['I'].max(),
    "epidemic-size" : lambda df: df['R'].iloc[-1]
}