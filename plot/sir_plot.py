import matplotlib.pyplot as plt
import numpy as np

for file in glob.glob('../simulations/Results/'+directory_name+'/SIR/*'):
    arr = np.load(file)
    plt.plot(arr[:,0],c='C0',alpha=.5)
    plt.plot(arr[:,1],c='C1',alpha=.5)
    plt.plot(arr[:,2],c='C2',alpha=.5)
plt.show()
