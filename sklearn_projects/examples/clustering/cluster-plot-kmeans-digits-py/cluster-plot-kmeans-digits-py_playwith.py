# panel 1
import numpy as np

from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)
print(f"data size: {data.shape}")
# print(f"data: {len(data)}, labels: {len(labels)}")
# print(f"data: {data[:3, :3]}, labels: {labels[:3]}")
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

# print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# panel 2
from time import time

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    if str(type(kmeans)) != "<class 'sklearn.cluster._kmeans.KMeans'>":
        raise Exception("wrong type (supposed to be KMeans)")

    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    # print(f"fit_time: {fit_time}, name: {name}")
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # ---
    # print(estimator[-1].__dir__())
    # print(estimator[-1].n_clusters)  # 10
    # # # print(estimator[-1].init)  # k-means++
    # print(estimator[-1].max_iter)  # 300
    # print(estimator[-1].tol)  # 0.0001
    # print(estimator[-1].n_init)  # 4
    # print(estimator[-1].verbose)  # 0
    # print(estimator[-1].random_state)  # 0
    # print(estimator[-1].copy_x)  # True
    # print(estimator[-1].algorithm)  # lloyd
    # print(estimator[-1].n_features_in_)  # 64
    # print(estimator[-1]._tol)  # _tol
    # print(estimator[-1]._n_init)  # _n_init
    # print(estimator[-1]._algorithm)  # _algorithm
    # print(estimator[-1]._n_threads)  # _n_threads
    # # # print(estimator[-1].cluster_centers_)  # cluster_centers_
    # print(estimator[-1]._n_features_out)  # _n_features_out
    # print(estimator[-1].labels_)  # labels_
    # print(estimator[-1].inertia_)  # inertia_
    # print(estimator[-1].n_iter_)  # n_iter_
    # # # raise Exception("stop here!!")
    # ---

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    # print(f"result (name, fit_time & estimator): {results}")

# panel 3
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
# print(f"type(kmeans): {type(kmeans)}")
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")

# panel 4
import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
print(f"reduced_data: ", reduced_data.shape)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# ---
print(f"x_min: {x_min}")
print(f"x_max: {x_max}")
print(f"y_min: {y_min}")
print(f"y_max: {y_max}")

print(f"xx: {xx.shape}")
print(f"yy: {yy.shape}")
# ---

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# plt.show()


# comments!!
comment1 = """
1) estimator[-1].__dir__()
['n_clusters', 'init', 'max_iter', 'tol', 'n_init', 'verbose', 'random_state', 'copy_x', 'algorithm', 'n_features_in_', '_tol', '_n_init', '_algorithm', '_n_threads', 'cluster_centers_', '_n_features_out', 'labels_', 'inertia_', 'n_iter_', '__module__', '__annotations__', '__doc__', '_parameter_constraints', '__init__', '_check_params_vs_input', '_warn_mkl_vcomp', 'fit', 'set_fit_request', 'set_score_request', '_sklearn_auto_wrap_output_keys', '__abstractmethods__', '_abc_impl', '_check_mkl_vcomp', '_validate_center_shape', '_check_test_data', '_init_centroids', 'fit_predict', 'predict', 'fit_transform', 'transform', '_transform', 'score', '_more_tags', 'get_feature_names_out', '__dict__', '__weakref__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__', 'set_output', '_estimator_type', '_get_param_names', 'get_params', 'set_params', '__sklearn_clone__', '__setstate__', '_get_tags', '_check_n_features', '_check_feature_names', '_validate_data', '_validate_params', '_repr_html_', '_repr_html_inner', '_repr_mimebundle_', '_doc_link_module', '_doc_link_url_param_generator', '_doc_link_template', '_get_doc_link', '_build_request_for_signature', '_get_default_requests', '_get_metadata_request', 'get_metadata_routing', '__slots__']

"""
