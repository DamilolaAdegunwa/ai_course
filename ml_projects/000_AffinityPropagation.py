import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

# Sample data (5 points in 2D space)
X = np.array([
    [1, 2],
    [2, 3],
    [5, 6],
    [8, 8],
    [8, 9]
])

# Create and fit the AffinityPropagation model
af = AffinityPropagation(random_state=42)
af.fit(X)

# Get the cluster centers and labels
cluster_centers = af.cluster_centers_
labels = af.labels_

# Print results
print("Cluster Centers:\n", cluster_centers)
print("Labels:", labels)
print("damping:", af.damping)
print("max_iter:", af.max_iter)
print("preference:", af.preference)
print("affinity:", af.affinity)
print("random_state:", af.random_state)
print("n_features_in_:", af.n_features_in_)
print("affinity_matrix_:", af.affinity_matrix_)
print("cluster_centers_indices_:", af.cluster_centers_indices_)
print("n_iter_:", af.n_iter_)
print("__module__:", af.__module__)
print("__annotations__:", af.__annotations__)
# print("__doc__:", af.__doc__)
# print("__doc__:", af.__dir__)
print("_parameter_constraints:", af._parameter_constraints)
print("__init__:", af.__init__())
print("__init__:", af._more_tags)
print("_estimator_type:", af._estimator_type)
print("__dict__:", af.__dict__)
print("__weakref__:", af.__weakref__)
print("__new__:", af.__new__)
print("__repr__:", af.__repr__())
print("__hash__:", af.__hash__())
"""
dir: ['damping', 'max_iter', 'convergence_iter', 'copy', 'verbose', 'preference', 'affinity', 'random_state', 'n_features_in_', 'affinity_matrix_', 'cluster_centers_indices_', 'labels_', 'n_iter_', 'cluster_centers_', '__module__', '__annotations__', '__doc__', '_parameter_constraints', '__init__', '_more_tags', 'fit', 'predict', 'fit_predict', '_estimator_type', '__dict__', '__weakref__', '__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__', '_get_param_names', 'get_params', 'set_params', '__sklearn_clone__', '__setstate__', '_get_tags', '_check_n_features', '_check_feature_names', '_validate_data', '_validate_params', '_repr_html_', '_repr_html_inner', '_repr_mimebundle_', '_doc_link_module', '_doc_link_url_param_generator', '_doc_link_template', '_get_doc_link', '_build_request_for_signature', '_get_default_requests', '_get_metadata_request', 'get_metadata_routing']

"""
