import numpy as np

data = np.load('graph_save/02/00161.npz')
data = np.load('graph_save/umn/test01/00161.npz')
print(data['f'].shape)