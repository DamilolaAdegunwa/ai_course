import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

X = np.array([[18.34,  0.77, -0.36, 16.99, 17.09],
              [-0.15, 65.74, 62.65,  0.09,  3.11],
              [-0.68, 61.49, 64.3, -0.84,  0.84],
              [17.68,  0.11, -0.77, 18.43, 15.81],
              [ 0.59, 62.3, 62.48, -0.66,  1.74]])

model = cluster.SpectralCoclustering(n_clusters=2)
model.fit(X)

fit_X = X[np.argsort(model.row_labels_)]
fit_X = fit_X[:, np.argsort(model.column_labels_)]

print(fit_X)
# plt.matshow(fit_X, cmap=plt.cm.Reds)
