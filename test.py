import numpy as np

a = np.load('mi_all_4_12345.npy')
b = np.load('train_4_12345.npy')
c = np.load('binned_entropy_4_12345.npy')
d = np.load('cluster_distances_4_12345.npy')

print(a.shape, b.shape, c.shape, d.shape)