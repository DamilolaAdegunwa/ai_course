from __future__ import annotations

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas.core.generic import NDFrame
from ucimlrepo import fetch_ucirepo
# ----------------------------------

iris_data_features: pd.DataFrame = (fetch_ucirepo(id=53)).data.features
sample_data_features: pd.DataFrame = pd.DataFrame(
[
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.9,  1.4, 0.2],
    [4.6, 3.1, 2.3, 1.2],
    [3.6, 2.1, 2.5, 2.2],
    [4.8, 3.3, 1.4, 3.2],
    [4.7, 3.7, -3.5, -0.2],
    [4.6, 3.1, -4.5, -9.2],
    [5.3,  8.6, 1.4, 0.2]
])
idf = iris_data_features.__deepcopy__()
# ---
# print("array***\n", np.array(idf))
print("__add__***\n", (sample_data_features.__add__(sample_data_features)))
# ---
# ['sepal length' 'sepal width' 'petal length' 'petal width']

# print("***columns description***")
# print(iris_data_features['sepal length'].describe())
# print(iris_data_features["sepal width"].describe())
# print(iris_data_features["petal length"].describe())
# print(iris_data_features["petal width"].describe())

# ---
# print("***columns ***")
# print(abs(iris_data_features['sepal length']))


# Convert dataset to a NumPy array for visualization
data: np.ndarray = iris_data_features.to_numpy()
# print(data)

# Visualizing a subset of the dataset as a heatmap
plt.matshow(data[:2, :4], cmap=plt.cm.twilight_shifted)  # Limiting to 300x300 for visualization
plt.title("Iris Data Visualization")
plt.colorbar()
# plt.show()

comment = """
# mice_protein = preprocessing.scale(mice_protein)
print("mice_protein standard: ", mice_protein)
# Extract data
X: pd.DataFrame = mice_protein.data.features  # Features
print("X.columns: = ", X.columns)
y: pd.Series = mice_protein.data.targets      # Target labels

# Convert dataset to a NumPy array for visualization
data: np.ndarray = X.to_numpy()

# Visualizing a subset of the dataset as a heatmap
plt.matshow(data[:300, :300], cmap=plt.cm.Blues)  # Limiting to 300x300 for visualization
"""

comment2 = """
 ['T', '_AXIS_LEN', '_AXIS_ORDERS', '_AXIS_TO_AXIS_NUMBER', '_HANDLED_TYPES', '__abs__', '__add__', '__and__', '__annotations__', '__array__', '__array_priority__', '__array_ufunc__', '__arrow_c_stream__', '__bool__', '__class__', '__contains__', '__copy__', '__dataframe__', '__dataframe_consortium_standard__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__divmod__', '__doc__', '__eq__', '__finalize__', '__floordiv__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__imod__', '__imul__', '__init__', '__init_subclass__', '__invert__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pandas_priority__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__', '__xor__', '_accessors', '_accum_func', '_agg_examples_doc', '_agg_see_also_doc', '_align_for_op', '_align_frame', '_align_series', '_append', '_arith_method', '_arith_method_with_reindex', '_as_manager', '_attrs', '_box_col_values', '_can_fast_transpose', '_check_inplace_and_allows_duplicate_labels', '_check_is_chained_assignment_possible', '_check_label_or_level_ambiguity', '_check_setitem_copy', '_clear_item_cache', '_clip_with_one_bound', '_clip_with_scalar', '_cmp_method', '_combine_frame', '_consolidate', '_consolidate_inplace', '_construct_axes_dict', '_construct_result', '_constructor', '_constructor_from_mgr', '_constructor_sliced', '_constructor_sliced_from_mgr', '_create_data_for_split_and_tight_to_dict', '_data', '_deprecate_downcast', '_dir_additions', '_dir_deletions', '_dispatch_frame_op', '_drop_axis', '_drop_labels_or_levels', '_ensure_valid_index', '_find_valid_index', '_flags', '_flex_arith_method', '_flex_cmp_method', '_from_arrays', '_from_mgr', '_get_agg_axis', '_get_axis', '_get_axis_name', '_get_axis_number', '_get_axis_resolvers', '_get_block_manager_axis', '_get_bool_data', '_get_cleaned_column_resolvers', '_get_column_array', '_get_index_resolvers', '_get_item_cache', '_get_label_or_level_values', '_get_numeric_data', '_get_value', '_get_values_for_csv', '_getitem_bool_array', '_getitem_multilevel', '_getitem_nocopy', '_getitem_slice', '_gotitem', '_hidden_attrs', '_indexed_same', '_info_axis', '_info_axis_name', '_info_axis_number', '_info_repr', '_init_mgr', '_inplace_method', '_internal_names', '_internal_names_set', '_is_copy', '_is_homogeneous_type', '_is_label_or_level_reference', '_is_label_reference', '_is_level_reference', '_is_mixed_type', '_is_view', '_is_view_after_cow_rules', '_iset_item', '_iset_item_mgr', '_iset_not_inplace', '_item_cache', '_iter_column_arrays', '_ixs', '_logical_func', '_logical_method', '_maybe_align_series_as_frame', '_maybe_cache_changed', '_maybe_update_cacher', '_metadata', '_mgr', '_min_count_stat_function', '_needs_reindex_multi', '_pad_or_backfill', '_protect_consolidate', '_reduce', '_reduce_axis1', '_reindex_axes', '_reindex_multi', '_reindex_with_indexers', '_rename', '_replace_columnwise', '_repr_data_resource_', '_repr_fits_horizontal_', '_repr_fits_vertical_', '_repr_html_', '_repr_latex_', '_reset_cache', '_reset_cacher', '_sanitize_column', '_series', '_set_axis', '_set_axis_name', '_set_axis_nocheck', '_set_is_copy', '_set_item', '_set_item_frame_value', '_set_item_mgr', '_set_value', '_setitem_array', '_setitem_frame', '_setitem_slice', '_shift_with_freq', '_should_reindex_frame_op', '_slice', '_stat_function', '_stat_function_ddof', '_take_with_is_copy', '_to_dict_of_blocks', '_to_latex_via_styler', '_typ', '_update_inplace', '_validate_dtype', '_values', '_where', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all', 'any', 'apply', 'applymap', 'asfreq', 'asof', 'assign', 'astype', 'at', 'at_time', 'attrs', 'axes', 'backfill', 'between_time', 'bfill', 'bool', 'boxplot', 'clip', 'columns', 'combine', 'combine_first', 'compare', 'convert_dtypes', 'copy', 'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'div', 'divide', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna', 'dtypes', 'duplicated', 'empty', 'eq', 'equals', 'eval', 'ewm', 'expanding', 'explode', 'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'flags', 'floordiv', 'from_dict', 'from_records', 'ge', 'get', 'groupby', 'gt', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'info', 'insert', 'interpolate', 'isetitem', 'isin', 'isna', 'isnull', 'items', 'iterrows', 'itertuples', 'join', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lt', 'map', 'mask', 'max', 'mean', 'median', 'melt', 'memory_usage', 'merge', 'min', 'mod', 'mode', 'mul', 'multiply', 'ndim', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique', 'pad', 'pct_change', 'pipe', 'pivot', 'pivot_table', 'plot', 'pop', 'pow', 'prod', 'product', 'quantile', 'query', 'radd', 'rank', 'rdiv', 'reindex', 'reindex_like', 'rename', 'rename_axis', 'reorder_levels', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'select_dtypes', 'sem', 'set_axis', 'set_flags', 'set_index', 'shape', 'shift', 'size', 'skew', 'sort_index', 'sort_values', 'squeeze', 'stack', 'std', 'style', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dict', 'to_excel', 'to_feather', 'to_gbq', 'to_hdf', 'to_html', 'to_json', 'to_latex', 'to_markdown', 'to_numpy', 'to_orc', 'to_parquet', 'to_period', 'to_pickle', 'to_records', 'to_sql', 'to_stata', 'to_string', 'to_timestamp', 'to_xarray', 'to_xml', 'transform', 'transpose', 'truediv', 'truncate', 'tz_convert', 'tz_localize', 'unstack', 'update', 'value_counts', 'values', 'var', 'where', 'xs']



 (<class 'pandas.core.series.Series'>, <class 'pandas.core.indexes.base.Index'>, <class 'pandas.core.arrays.base.ExtensionArray'>, <class 'numpy.ndarray'>)
"""
