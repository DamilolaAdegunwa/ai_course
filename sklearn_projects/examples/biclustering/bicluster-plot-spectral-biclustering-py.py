# 1
import pandas
from matplotlib import pyplot as plt

from sklearn.datasets import make_checkerboard

# simulated data
n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
)

data: pandas.DataFrame = pandas.DataFrame(data)
print("\n\ninit data: \n", data)



# plot the graph
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")
# _ = plt.show()

#  2
import numpy as np

# Creating lists of shuffled row and column indices
rng = np.random.RandomState(0)
# data = rng.permutation(data.shape[0])
print("\n\nrow shuffled data: \n", data)

row_idx_shuffled = rng.permutation(data.shape[0]);
# print("row_idx_shuffled", row_idx_shuffled, "\nlength\n", len(row_idx_shuffled),"\n\n\n")
col_idx_shuffled = rng.permutation(data.shape[1]);
# print("col_idx_shuffled", col_idx_shuffled, "\nlength\n", len(col_idx_shuffled),"\n\n\n")

# print("\n\ninit data.shape: \n", len(data.shape))

#  3
data = data[row_idx_shuffled][:, col_idx_shuffled]; # print("\n\ndata: \n", data)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")
# _ = plt.show()


#  4
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
model.fit(data)

# Compute the similarity of two sets of biclusters
score = consensus_score(
    model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled])
)
print(f"consensus score: {score:.1f}")


#  5
# Reordering first the rows and then the columns.
reordered_rows = data[np.argsort(model.row_labels_)]
reordered_data = reordered_rows[:, np.argsort(model.column_labels_)]

plt.matshow(reordered_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")
#  _ = plt.show()


#  6
plt.matshow(
    np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")
# plt.show()
