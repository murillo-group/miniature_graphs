"""
Random Graph Sampling functions for Miniaturize
"""
import time
import tracemalloc
import csv

def profile(csvfile=open("log.txt","w",newline='')):
    stream = csv.writer(csvfile,delimiter=',')

    def decorate(func):
        def wrapper(*args,**kwargs):
            # Evaluate and time function
            start = time.time()
            tracemalloc.start()
            result = func(*args,**kwargs)
            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            stop = time.time()

            # Write to csv file
            stream.writerow([func.__name__, 
                            f"{stop-start:.4f}",
                            mem[1] * 1e-6,
                            *args,
                            ])

            return result
        return wrapper
    return decorate