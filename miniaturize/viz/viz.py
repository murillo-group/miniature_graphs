# Import modules
import matplotlib.pyplot as plt
import numpy as np
import functools

# Define environment variables
STYLE_SHEET_DIR=""

def labels(axs):
    if np.shape(axs) == ():
        axs.set_title('Y-Axis vs. X-Axis')
        axs.set_xlabel('X-Axis')
        axs.set_ylabel('Y-Axis')
    else:
        for ax in axs: labels(ax)

def label(func):
    @functools.wraps(func)
    def labelled(*args,**kwargs):
        fig, axs = func(*args,**kwargs)

        # Label each axes individually
        labels(axs)

        return fig, axs

    return labelled

def initialize(path=None):
    def apply(func,dec):
        if not hasattr(func,'__wrapped__'):
            func = dec(func)
        
        return func
    
    if not None:
        plt.style.use(path)

    plt.subplots = apply(plt.subplots,label)

def finalize():
    def remove(func):
        if hasattr(func,'__wrapped__'):
            func = func.__wrapped__

        return func
    
    plt.subplots = remove(plt.subplots)
