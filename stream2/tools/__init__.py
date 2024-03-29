"""The core functionality."""

from ._dimension_reduction import dimension_reduction
from ._elpigraph import learn_graph, seed_graph
from ._pseudotime import infer_pseudotime
from ._markers import (
    detect_transition_markers,
    spearman_columns,
    spearman_pairwise,
    xicorr_columns,
    xicorr_pairwise,
)
from ._graph_utils import (
    add_path,
    del_path,
    find_paths,
    refit_graph,
    extend_leaves,
    prune_graph,
    get_weights,
    get_component,
    find_disconnected_components,
    ordinal_knn,
    smooth_ordinal_labels,
    early_groups,
    interpolate,
    use_graph_with_n_nodes,
    project_graph
)
