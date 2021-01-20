import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples = 10, centers = 4, n_features=3, shuffle=True, random_state=31)
x = np.array([[1, 2, 3], [1, 2, 2.5], [1, 3, 2.5], [50, 52, 52.5], [58, 52, 52.5], [50, 52, 52.5], [100, 200, 150],
              [150, 220, 180]])
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x,4, 2, error=0.005, maxiter=1000, init=None)

print('cntr', cntr)
print('u', u)
print('u0', u0)
print('jm', jm)

cluster_membership = np.argmax(u, axis=0)
print('cluster_membership', cluster_membership)