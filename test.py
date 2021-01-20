# import numpy as np
# import cmeans
# from matplotlib import pyplot as plt
#
#
# n_samples = 3000
#
# X = np.concatenate((
#     np.random.normal((-2, -2), size=(n_samples, 2)),
#     np.random.normal((2, 2), size=(n_samples, 2))
# ))
#
# fcm = cmeans.FCM(n_clusters=2)
# fcm.fit(X)