import networkx as nx
import elpigraph
import numpy as np
import scanpy as sc
import scipy
import copy
import statsmodels.api
from sklearn.neighbors import KNeighborsRegressor

from ._elpigraph import (
    learn_graph,
    _store_graph_attributes,
    _get_graph_data,
    _subset_adata,
)
from .._utils import stream2elpi


def project_graph(adata, to_basis="X_umap", key="epg"):

    obsm = adata.uns[key]["params"]["obsm"]
    layer = adata.uns[key]["params"]["layer"]
    if obsm is not None:
        X = adata.obsm[obsm].copy()
        from_basis = obsm
    elif layer is not None:
        X = adata.layers[layer].copy()
        from_basis = layer
    else:
        X = adata.X
        from_basis = 'X'

    suffix = f"_from_{from_basis}_to_{to_basis}"
    adata.uns[key + suffix] = copy.deepcopy(adata.uns[key])
    adata.uns[key + suffix]["params"]["obsm"] = "X_umap"

    # proj
    adata.uns[key + suffix]["node_pos"] = elpigraph.utils.proj2embedding(
        X,
        adata.obsm[to_basis],
        adata.uns[key]["node_pos"],
    )
    empty_nodes = np.where(
        np.isnan(adata.uns[key + suffix]["node_pos"][:, 0])
    )[0]
    for node in empty_nodes:
        neigh_nodes = adata.uns[key + suffix][
            "node_pos"
        ][
            np.unique(
                adata.uns[key + suffix]["edge"][
                    (adata.uns[key + suffix]["edge"] == node).any(axis=1)
                ]
            )
        ]
        adata.uns[key + suffix]["node_pos"][node] = np.nanmean(
            neigh_nodes, axis=0
        )


def find_paths(
    adata,
    min_path_len=None,
    n_nodes=None,
    max_inner_fraction=0.1,
    min_node_n_points=None,
    max_n_points=None,
    min_compactness=0.5,
    radius=None,
    allow_same_branch=True,
    fit_loops=True,
    plot=False,
    verbose=False,
    inplace=False,
    use_weights=False,
    use_partition=False,
    epg_lambda=None,
    epg_mu=None,
    epg_cycle_lambda=None,
    epg_cycle_mu=None,
    ignore_equivalent=False,
    key="epg",
):
    """This function tries to add extra paths to the graph by computing a
    series of principal curves connecting two nodes and retaining plausible
    ones using heuristic parameters.

    min_path_len: int, default=None
        Minimum distance along the graph (in number of nodes)
        that separates the two nodes to connect with a principal curve
    n_nodes: int, default=None
        Number of nodes in the candidate principal curves
    max_inner_fraction: float in [0,1], default=0.1
        Maximum fraction of points inside vs outside the loop
        (controls how empty the loop formed with the added path should be)
    min_node_n_points: int, default=1
        Minimum number of points associated to nodes of the principal curve
        (prevents creating paths through empty space)
    max_n_points: int, default=5% of the number of points
        Maximum number of points inside the loop
    min_compactness: float in [0,1], default=0.5
        Minimum 'roundness' of the loop (1=more round)
        (if very narrow loops are not desired)
    radius: float, default=None
        Max distance in space that separates
        the two nodes to connect with a principal curve
    allow_same_branch: bool, default=True
        Whether to allow new paths to connect two nodes from the same branch
    fit_loops: bool, default=True
        Whether to refit the graph to data after adding the new paths
    plot: bool, default=False
        Whether to plot selected candidate paths
    verbose: bool, default=False
    copy: bool, default=False
    use_weights: bool, default=False
        Whether to use point weights
    use_partition: bool or list, default=False
    """
    if use_partition:
        if verbose:
            print("Searching potential loops for each partition...")
        if type(use_partition) is bool:
            partitions = adata.obs["partition"].unique()
        elif type(use_partition) is list:
            partitions = use_partition
        else:
            raise ValueError(
                "use_partition should be a bool" + "or a list of partitions"
            )

        merged_nodep = []
        merged_edges = []
        num_edges = 0
        for part in adata.obs["partition"].unique():

            p_adata = _subset_adata(adata, part)
            if part in partitions:
                _find_paths(
                    p_adata,
                    min_path_len=min_path_len,
                    n_nodes=n_nodes,
                    max_inner_fraction=max_inner_fraction,
                    min_node_n_points=min_node_n_points,
                    max_n_points=max_n_points,
                    min_compactness=min_compactness,
                    radius=radius,
                    allow_same_branch=allow_same_branch,
                    fit_loops=fit_loops,
                    epg_lambda=epg_lambda,
                    epg_mu=epg_mu,
                    epg_cycle_lambda=epg_cycle_lambda,
                    epg_cycle_mu=epg_cycle_mu,
                    use_weights=use_weights,
                    ignore_equivalent=ignore_equivalent,
                    plot=plot,
                    verbose=verbose,
                    inplace=inplace,
                    key=key,
                )

            merged_nodep.append(p_adata.uns[key]["node_pos"])
            merged_edges.append(p_adata.uns[key]["edge"] + num_edges)
            num_edges += len(p_adata.uns[key]["node_pos"])

        adata.uns[key] = {}
        adata.uns[key]["node_pos"] = np.concatenate(merged_nodep)
        adata.uns[key]["edge"] = np.concatenate((merged_edges))
        adata.uns[key]["node_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(nodep) for nodep in merged_nodep],
        ).astype(str)
        adata.uns[key]["edge_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(edges) for edges in merged_edges],
        ).astype(str)
        adata.uns[key]["params"] = p_adata.uns[key]["params"]

        X = _get_graph_data(adata, key=key)
        _store_graph_attributes(adata, X, key=key)

    else:
        if verbose:
            print("Searching potential loops...")

        _find_paths(
            adata,
            min_path_len=min_path_len,
            n_nodes=n_nodes,
            max_inner_fraction=max_inner_fraction,
            min_node_n_points=min_node_n_points,
            max_n_points=max_n_points,
            min_compactness=min_compactness,
            radius=radius,
            allow_same_branch=allow_same_branch,
            fit_loops=fit_loops,
            epg_lambda=epg_lambda,
            epg_mu=epg_mu,
            epg_cycle_lambda=epg_cycle_lambda,
            epg_cycle_mu=epg_cycle_mu,
            use_weights=use_weights,
            ignore_equivalent=ignore_equivalent,
            plot=plot,
            verbose=verbose,
            inplace=inplace,
            key=key,
        )


def _find_paths(
    adata,
    min_path_len=None,
    n_nodes=None,
    max_inner_fraction=0.1,
    min_node_n_points=None,
    max_n_points=None,
    min_compactness=0.5,
    radius=None,
    allow_same_branch=True,
    fit_loops=True,
    epg_lambda=None,
    epg_mu=None,
    epg_cycle_lambda=None,
    epg_cycle_mu=None,
    use_weights=False,
    ignore_equivalent=False,
    plot=False,
    verbose=True,
    inplace=False,
    key="epg",
):

    # --- Init parameters, variables
    if use_weights:
        if "pointweights" not in adata.obs:
            raise ValueError(
                "adata.obs['pointweights'] missing. Run st2.tl.get_weights"
            )
        weights = np.array(adata.obs["pointweights"]).reshape((-1, 1))
    else:
        weights = None

    X = _get_graph_data(adata, key)
    PG = stream2elpi(adata, key)
    PG = elpigraph.findPaths(
        X,
        PG,
        Mu=epg_mu,
        Lambda=epg_lambda,
        cycle_Lambda=epg_cycle_lambda,
        cycle_Mu=epg_cycle_mu,
        min_path_len=min_path_len,
        nnodes=n_nodes,
        max_inner_fraction=max_inner_fraction,
        min_node_n_points=min_node_n_points,
        max_n_points=max_n_points,
        min_compactness=min_compactness,
        radius=radius,
        allow_same_branch=allow_same_branch,
        fit_loops=fit_loops,
        weights=weights,
        ignore_equivalent=ignore_equivalent,
        plot=plot,
        verbose=verbose,
    )

    if PG is None:
        return
    if inplace:
        adata.uns[key]["node_pos"] = PG["addLoopsdict"]["merged_nodep"]
        adata.uns[key]["edge"] = PG["addLoopsdict"]["merged_edges"]
        # update edge_len, conn, data projection
        _store_graph_attributes(adata, X, key)


def add_path(
    adata,
    source,
    target,
    n_nodes=None,
    use_weights=False,
    refit_graph=False,
    epg_mu=None,
    epg_lambda=None,
    epg_cycle_mu=None,
    epg_cycle_lambda=None,
    key="epg",
):

    # --- Init parameters, variables
    if epg_mu is None:
        epg_mu = adata.uns[key]["params"]["epg_mu"]
    if epg_lambda is None:
        epg_lambda = adata.uns[key]["params"]["epg_lambda"]
    if epg_cycle_mu is None:
        epg_cycle_mu = epg_mu
    if epg_cycle_lambda is None:
        epg_cycle_lambda = epg_lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    X = _get_graph_data(adata, key)
    PG = stream2elpi(adata, key)
    PG = elpigraph.addPath(
        X,
        PG=PG,
        source=source,
        target=target,
        n_nodes=n_nodes,
        weights=weights,
        refit_graph=refit_graph,
        Mu=epg_mu,
        Lambda=epg_lambda,
        cycle_Mu=epg_cycle_mu,
        cycle_Lambda=epg_cycle_lambda,
    )

    adata.uns["epg"]["node_pos"] = PG["NodePositions"]
    adata.uns["epg"]["edge"] = PG["Edges"][0]

    # update edge_len, conn, data projection
    _store_graph_attributes(adata, X, key)


def del_path(
    adata,
    source,
    target,
    nodes_to_include=None,
    use_weights=False,
    refit_graph=False,
    epg_mu=None,
    epg_lambda=None,
    epg_cycle_mu=None,
    epg_cycle_lambda=None,
    key="epg",
):

    # --- Init parameters, variables
    if epg_mu is None:
        epg_mu = adata.uns[key]["params"]["epg_mu"]
    if epg_lambda is None:
        epg_lambda = adata.uns[key]["params"]["epg_lambda"]
    if epg_cycle_mu is None:
        epg_cycle_mu = epg_mu
    if epg_cycle_lambda is None:
        epg_cycle_lambda = epg_lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    X = _get_graph_data(adata, key)
    PG = stream2elpi(adata, key)
    PG = elpigraph.delPath(
        X,
        PG=PG,
        source=source,
        target=target,
        nodes_to_include=nodes_to_include,
        weights=weights,
        refit_graph=refit_graph,
        Mu=epg_mu,
        Lambda=epg_lambda,
        cycle_Mu=epg_cycle_mu,
        cycle_Lambda=epg_cycle_lambda,
    )

    adata.uns["epg"]["node_pos"] = PG["NodePositions"]
    adata.uns["epg"]["edge"] = PG["Edges"][0]

    # update edge_len, conn, data projection
    _store_graph_attributes(adata, X, key)


def prune_graph(
    adata,
    mode="PointNumber",
    collapse_par=5,
    trimming_radius=np.inf,
    refit_graph=False,
    copy=False,
):
    pg = {
        "NodePositions": adata.uns["epg"]["node_pos"].copy(),
        "Edges": [adata.uns["epg"]["edge"].copy()],
    }
    pg2 = elpigraph.CollapseBranches(
        adata.obsm["X_dr"],
        pg,
        ControlPar=collapse_par,
        Mode=mode,
        TrimmingRadius=trimming_radius,
    )
    if not copy:
        if refit_graph:
            params = adata.uns["epg"]["params"]
            learn_graph(
                adata,
                method=params["method"],
                obsm=params["obsm"],
                layer=params["layer"],
                n_nodes=params["n_nodes"],
                epg_lambda=params["epg_lambda"],
                epg_mu=params["epg_mu"],
                epg_alpha=params["epg_alpha"],
                use_seed=False,
                InitNodePositions=pg2["Nodes"],
                InitEdges=pg2["Edges"],
            )
        else:
            adata.uns["epg"]["node_pos"] = pg2["Nodes"]
            adata.uns["epg"]["edge"] = pg2["Edges"]
    else:
        return pg2


def find_disconnected_components(
    adata, groups="leiden", neighbors_key=None, verbose=True
):
    """Find if data contains disconnected components.

    Inputs
    ------
    adata : anndata.AnnData class instance

    Returns
    -------
    adata.obs['partition']: component assignment of points
    """

    if groups not in adata.obs:
        raise ValueError(f"{groups} not found in adata.obs")

    sc.tl.paga(adata, groups=groups, neighbors_key=neighbors_key)
    # edges = np.argwhere(adata.uns["paga"]["connectivities"])
    # edges_tree = np.argwhere(adata.uns["paga"]["connectivities_tree"])
    g = nx.convert_matrix.from_scipy_sparse_array(
        adata.uns["paga"]["connectivities_tree"]
    )
    comps = [list(c) for c in nx.algorithms.components.connected_components(g)]
    clus_idx = [
        np.where(adata.obs[adata.uns["paga"]["groups"]].astype(int) == i)[0]
        for i in g.nodes
    ]

    partition = np.zeros(len(adata), dtype=object)
    for i, comp in enumerate(comps):
        comp_idx = np.concatenate([clus_idx[i] for i in comp])
        partition[comp_idx] = str(i)
    adata.obs["partition"] = partition
    print(f"Found", len(adata.obs["partition"].unique()), "components")


def get_weights(
    adata,
    obsm="X_dr",
    layer=None,
    bandwidth=1,
    griddelta=100,
    exponent=1,
    method="sklearn",
    **kwargs,
):
    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm:
            mat = adata.obsm[obsm]
        else:
            raise ValueError(f"could not find {obsm} in `adata.obsm`")
    elif layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer]
        else:
            raise ValueError(f"could not find {layer} in `adata.layers`")
    else:
        mat = adata.X

    adata.obs["pointweights"] = elpigraph.utils.getWeights(
        mat, bandwidth, griddelta, exponent, method, **kwargs
    )


def get_component(adata, component):
    sadata = _subset_adata(adata, component)
    for key in ["seed_epg", "epg"]:
        if key in sadata.uns:
            X = _get_graph_data(sadata, "epg")
            _store_graph_attributes(sadata, X, key)
    return sadata


def ordinal_knn(
    adata,
    ordinal_label,
    obsm="X_pca",
    layer=None,
    n_neighbors=15,
    n_natural=1,
    metric="cosine",
    method="guide",
    return_sparse=False,
    stages=None,
):
    """Supervised (ordinal) nearest-neighbor search.

    Parameters
    ----------
    n_neighbors: int
        Number of neighbors
    n_natural: int
        Number of natural neighbors (between 0 and n_neighbors-1)
        to force the graph to retain. Tunes the strength of supervision
    metric: str
        One of sklearn's distance metrics
    method : str (default='force')
        if 'force', for each point at stage[i] get n_neighbors, forcing:
            - n_neighbors/3 to be from stage[i-1]
            - n_neighbors/3 to be from stage[i]
            - n_neighbors/3 to be from stage[i+1]
            For stage[0] and stage[-1], 2*n_neighbors/3 are taken from stage[i]

        if 'guide', for each point at stage[i] get n_neighbors
            from points in {stage[i-1], stage[i], stage[i+1]},
            without constraints on proportions
    return_sparse: bool
        Whether to return the graph in sparse form
        or as longform indices and distances
    stages: list
        Ordered list of ordinal label stages (low to high).
        If None, taken as np.unique(ordinal_label)

    Returns
    -------
    Supervised nearest-neighbors as a graph in sparse form
    or as longform indices and distances

    """

    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm:
            mat = adata.obsm[obsm]
        else:
            raise ValueError(f"could not find {obsm} in `adata.obsm`")
    elif layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer]
        else:
            raise ValueError(f"could not find {layer} in `adata.layers`")
    else:
        mat = adata.X

    out = elpigraph.utils.supervised_knn(
        mat,
        stages_labels=adata.obs[ordinal_label],
        stages=stages,
        method=method,
        n_neighbors=n_neighbors,
        n_natural=n_natural,
        m=metric,
        return_sparse=return_sparse,
    )

    if return_sparse:
        return out
    else:
        knn_dists, knn_idx = out
        return knn_dists, knn_idx


def smooth_ordinal_labels(
    adata,
    root,
    ordinal_label,
    obsm="X_pca",
    layer=None,
    n_neighbors=15,
    n_natural=1,
    metric="euclidean",
    method="guide",
    stages=None,
):
    """Smooth ordinal labels into a continuous vector

    Parameters
    ----------
    root: int
        Index of chosen root data points
    n_neighbors: int
        Number of neighbors
    n_natural: int
        Number of natural neighbors (between 0 and n_neighbors-1)
        to force the graph to retain. Tunes the strength of supervision
    metric: str
        One of sklearn's distance metrics
    method : str (default='force')
        if 'force', for each point at stage[i] get n_neighbors, forcing:
            - n_neighbors/3 to be from stage[i-1]
            - n_neighbors/3 to be from stage[i]
            - n_neighbors/3 to be from stage[i+1]
            For stage[0] and stage[-1], 2*n_neighbors/3 are taken from stage[i]

        if 'guide', for each point at stage[i] get n_neighbors
            from points in {stage[i-1], stage[i], stage[i+1]},
            without constraints on proportions
    return_sparse: bool
        Whether to return the graph in sparse form
        or as longform indices and distances
    stages: list
        Ordered list of ordinal label stages (low to high).
        If None, taken as np.unique(ordinal_label)

    Returns
    -------
    adata.obs['ps']: smoothed ordinal labels

    """
    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm:
            mat = adata.obsm[obsm]
        else:
            raise ValueError(f"could not find {obsm} in `adata.obsm`")
    elif layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer]
        else:
            raise ValueError(f"could not find {layer} in `adata.layers`")
    else:
        mat = adata.X

    g = elpigraph.utils.supervised_knn(
        mat,
        stages_labels=adata.obs[ordinal_label],
        stages=stages,
        n_natural=n_natural,
        n_neighbors=n_neighbors,
        m=metric,
        method=method,
        return_sparse=True,
    )

    adata.obs["ps"] = elpigraph.utils.geodesic_pseudotime(
        mat, n_neighbors, root=root, g=g
    )


def refit_graph(
    adata,
    use_weights=False,
    shift_nodes_pos={},
    epg_mu=None,
    epg_lambda=None,
    cycle_epg_mu=None,
    cycle_epg_lambda=None,
):
    """Refit graph to data

    Parameters
    ----------
    use_weights: bool
        Whether to weight points with adata.obs['pointweights']
    shift_nodes_pos: dict
        Optional dict to hold some nodes fixed at specified positions
        e.g., {2:[.5,.2]} will hold node 2 at coordinates [.5,.2]
    epg_mu: float
        ElPiGraph Mu parameter
    epg_lambda: float
        ElPiGraph Lambda parameter
    cycle_epg_mu: float
        ElPiGraph Mu parameter, specific for nodes that are part of cycles
    cycle_epg_lambda: float
        ElPiGraph Lambda parameter, specific for nodes that are part of cycles
    """
    # --- Init parameters, variables
    if epg_mu is None:
        epg_mu = adata.uns["epg"]["params"]["epg_mu"]
    if epg_lambda is None:
        epg_lambda = adata.uns["epg"]["params"]["epg_lambda"]
    if cycle_epg_mu is None:
        cycle_epg_mu = epg_mu
    if cycle_epg_lambda is None:
        cycle_epg_lambda = epg_lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    X = _get_graph_data(adata, "epg")
    PG = stream2elpi(adata, "epg")
    elpigraph._graph_editing.refitGraph(
        X,
        PG=PG,
        shift_nodes_pos=shift_nodes_pos,
        PointWeights=weights,
        Mu=epg_mu,
        Lambda=epg_lambda,
        cycle_Mu=cycle_epg_mu,
        cycle_Lambda=cycle_epg_lambda,
    )

    adata.uns["epg"]["node_pos"] = PG["NodePositions"]

    # update edge_len, conn, data projection
    _store_graph_attributes(adata, X, "epg")


def extend_leaves(
    adata,
    Mode="QuantDists",
    ControlPar=0.5,
    DoSA=True,
    DoSA_maxiter=200,
    LeafIDs=None,
    TrimmingRadius=float("inf"),
    key="epg",
):
    X = _get_graph_data(adata, key)

    PG = elpigraph.ExtendLeaves(
        X.astype(float),
        PG=stream2elpi(adata, key),
        Mode=Mode,
        ControlPar=ControlPar,
        DoSA=DoSA,
        DoSA_maxiter=DoSA_maxiter,
        LeafIDs=LeafIDs,
        TrimmingRadius=TrimmingRadius,
    )

    adata.uns[key]["node_pos"] = PG["NodePositions"]
    adata.uns[key]["edge"] = PG["Edges"][0]
    _store_graph_attributes(adata, X, key)


def use_graph_with_n_nodes(adata, n_nodes):
    """Use the graph at n_nodes.
    This requires having run st2.tl.learn_graph with store_evolution=True
    """

    adata.uns["epg"]["node_pos"] = adata.uns["epg"]["graph_evolution"][
        "all_node_pos"
    ][n_nodes]
    adata.uns["epg"]["edge"] = elpigraph.src.core.DecodeElasticMatrix2(
        adata.uns["epg"]["graph_evolution"]["all_edge"][n_nodes]
    )[0]
    adata.uns["epg"]["conn"] = scipy.sparse.csr_matrix(
        adata.uns["epg"]["graph_evolution"]["all_edge"][n_nodes]
    )
    X = _get_graph_data(adata, "epg")
    _store_graph_attributes(adata, X, "epg")


def early_groups(
    adata,
    branch_nodes,
    source,
    target,
    nodes_to_include=None,
    flavor="ot_unbalanced",
    n_windows=20,
    n_neighbors=20,
    ot_reg_e=0.01,
    ot_reg_m=0.001,
    key="epg",
):
    """
    Split data between source and target (with target a branching node)
    into n_windows slices along pseudotime.
    Then try to guess which branch the data prior
    to the branching most resembles.
    branch_nodes are adjacent to target and represent the separate branches.
    Labels are propagated back in pseudotime for each of the n_windows slices
    (e.g., from branch_nodes to slice[n_windows-1],
    then from slice[n_windows-1] to slice[n_windows-2],etc)

    Parameters
    ----------
    branch_nodes: list[int]
        List of node labels adjacent to target branch node
    source: int
        Root node label
    target: int
        Branching node label
    nodes_to_include: list[int]
        Nodes to include in the path between source and target
    flavor: str
        How to propagate labels from branch_nodes
        to the previous pseudotime slice
            "ot" for optimal transport
            "ot_unbalanced" for unbalanced OT
            "ot_equal" for OT with weight of each branch_nodes equalized
            "knn" for simple nearest-neighbor search
    n_windows: int
        How many slices along pseudotime to make
        with data between source and target
    n_neighbors: int
        Number of nearest neighbors for flavor=
    ot_reg_e: float
        Unbalanced optimal transport entropic regularization parameter
    ot_reg_m: float
        Unbalanced optimal transport unbalanced parameter
    key: str
        Graph key
    """
    X = _get_graph_data(adata, key)
    PG = stream2elpi(adata, key)
    elpigraph.utils.early_groups(
        X,
        PG,
        branch_nodes=branch_nodes,
        source=source,
        target=target,
        nodes_to_include=nodes_to_include,
        flavor=flavor,
        n_windows=n_windows,
        n_neighbors=n_neighbors,
        ot_reg_e=ot_reg_e,
        ot_reg_m=ot_reg_m,
    )

    s = "-".join(str(x) for x in branch_nodes)
    adata.obs[f"early_groups_{source}->{s}"] = PG[
        f"early_groups_{source}->{s}"
    ]
    adata.obs[f"early_groups_{source}->{s}_clusters"] = PG[
        f"early_groups_{source}->{s}_clusters"
    ]


def interpolate(
    adata,
    t_len=200,
    method="knn",
    frac=0.1,
    n_neighbors="auto",
    weights="uniform",
    key="epg",
):
    """Resample adata.X by interpolation along pseudotime with t_len values

    Parameters
    ----------
    t_len: int
        Number of pseudotime values to resample
    method: str
        'knn' for sklearn.neighbors.KNeighborsRegressor
        'lowess' for statsmodels.api.nonparametric.lowess (can be slow)
    frac: float 0-1
        lowess frac parameter
    n_neighbors: int
        KNeighborsRegressor n_neighbors parameter
    weights: str, 'uniform' or 'distance'
        KNeighborsRegressor weights parameter.

    Returns
    -------
    t_new: np.array
        Resampled pseudotime values
    interp: np.array
        Resampled adata.X
    """

    X = adata.X
    pseudotime = adata.obs[f"{key}_pseudotime"]

    idx_path = ~np.isnan(pseudotime)
    X_path = X[idx_path]

    t_path = np.array(pseudotime[idx_path]).reshape(-1, 1)
    t_new = np.linspace(pseudotime.min(), pseudotime.max(), t_len).reshape(
        -1, 1
    )

    if method == "knn":
        if n_neighbors == "auto":
            n_neighbors = int(len(X_path) * 0.05)
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        interp = reg.fit(X=t_path, y=X_path).predict(t_new)

    elif method == "lowess":  # very slow
        interp = np.zeros((t_len, X_path.shape[1]))
        for i in range(X_path.shape[1]):
            interp[:, i] = statsmodels.api.nonparametric.lowess(
                X_path[:, i],
                t_path.flat,
                it=1,
                frac=frac,
                xvals=t_new.flat,
                return_sorted=False,
            )
    else:
        raise ValueError("method must be one of 'knn','lowess'")
    return t_new, interp
