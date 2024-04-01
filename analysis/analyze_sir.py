import numpy as np
import matplotlib.pyplot as plt
import glob

print(glob.glob('../simulations/Results/example_graph/graph.npz_SIR/*'))

for file in glob.glob('../simulations/Results/example_graph/graph.npz_SIR/*'):
    arr = np.load(file)
    plt.plot(arr[:,0],c='C0',alpha=.5)
    plt.plot(arr[:,1],c='C1',alpha=.5)
    plt.plot(arr[:,2],c='C2',alpha=.5)
plt.show()

# arr = np.load('../simulations/Results/example_graph/graph.npz_SIR/run_2024_04_01_12_24_16_240273.npy')


# plt.show()
