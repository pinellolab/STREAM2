import networkx as nx
import elpigraph
import numpy as np
import scanpy as sc
import pandas as pd
from shapely.geometry import MultiLineString, LineString

from ._elpigraph import (
    learn_graph,
    _store_graph_attributes,
    _get_graph_data,
    _subset_adata,
)
from .._utils import stream2elpi


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
                    Lambda=p_adata.uns[key]["params"]["epg_lambda"],
                    Mu=p_adata.uns[key]["params"]["epg_mu"],
                    use_weights=use_weights,
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
            Lambda=adata.uns[key]["params"]["epg_lambda"],
            Mu=adata.uns[key]["params"]["epg_mu"],
            use_weights=use_weights,
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
    Lambda=0.02,
    Mu=0.1,
    use_weights=False,
    plot=False,
    verbose=True,
    inplace=False,
    key="epg",
):

    # --- Init parameters, variables
    X = _get_graph_data(adata, key)
    init_nodes_pos = adata.uns[key]["node_pos"]
    init_edges = adata.uns[key]["edge"]

    if use_weights:
        if "pointweights" not in adata.obs:
            raise ValueError(
                "adata.obs['pointweights'] missing. Run st2.tl.get_weights"
            )
        weights = np.array(adata.obs["pointweights"]).reshape((-1, 1))
    else:
        weights = None

    PG = elpigraph.findPaths(
        X,
        dict(
            NodePositions=init_nodes_pos,
            Edges=[init_edges],
            Lambda=Lambda,
            Mu=Mu,
        ),
        min_path_len=min_path_len,
        nnodes=n_nodes,
        max_inner_fraction=max_inner_fraction,
        min_node_n_points=min_node_n_points,
        max_n_points=max_n_points,
        # max_empty_curve_fraction=.2,
        min_compactness=min_compactness,
        radius=radius,
        allow_same_branch=allow_same_branch,
        fit_loops=fit_loops,
        weights=weights,
        plot=plot,
        verbose=verbose,
    )

    if PG is None:
        print("Found no valid path to add")
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
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
    key="epg",
):

    X = _get_graph_data(adata, key)

    # --- Init parameters, variables
    if Mu is None:
        Mu = adata.uns[key]["params"]["epg_mu"]
    if Lambda is None:
        Lambda = adata.uns[key]["params"]["epg_lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    PG = stream2elpi(adata, key)
    PG["projection"] = {}
    PG["projection"]["edge_len"] = adata.uns[key]["edge_len"]
    PG = elpigraph.addPath(
        X,
        PG=PG,
        source=source,
        target=target,
        n_nodes=n_nodes,
        weights=weights,
        refit_graph=refit_graph,
        Mu=Mu,
        Lambda=Lambda,
        cycle_Mu=cycle_Mu,
        cycle_Lambda=cycle_Lambda,
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
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
    key="epg",
):

    X = _get_graph_data(adata, key)

    # --- Init parameters, variables
    if Mu is None:
        Mu = adata.uns[key]["params"]["epg_mu"]
    if Lambda is None:
        Lambda = adata.uns[key]["params"]["epg_lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    PG = stream2elpi(adata, key)
    PG["projection"] = {}
    PG["projection"]["edge_len"] = adata.uns[key]["edge_len"]
    PG = elpigraph.delPath(
        X,
        PG=PG,
        source=source,
        target=target,
        nodes_to_include=nodes_to_include,
        weights=weights,
        refit_graph=refit_graph,
        Mu=Mu,
        Lambda=Lambda,
        cycle_Mu=cycle_Mu,
        cycle_Lambda=cycle_Lambda,
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
    g = nx.convert_matrix.from_scipy_sparse_matrix(
        adata.uns["paga"]["connectivities_tree"]
    )
    comps = [
        list(c) for c in nx.algorithms.components.connected_components(g)
    ]
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
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
):

    X = _get_graph_data(adata, "epg")
    init_nodes_pos, init_edges = (
        adata.uns["epg"]["node_pos"],
        adata.uns["epg"]["edge"],
    )

    # --- Init parameters, variables
    if Mu is None:
        Mu = adata.uns["epg"]["params"]["epg_mu"]
    if Lambda is None:
        Lambda = adata.uns["epg"]["params"]["epg_lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    PG = {
        "NodePositions": adata.uns["epg"]["node_pos"].astype(float),
        "Edges": [adata.uns["epg"]["edge"]],
    }
    elpigraph._graph_editing.refitGraph(
        X,
        PG=PG,
        shift_nodes_pos=shift_nodes_pos,
        PointWeights=weights,
        Mu=Mu,
        Lambda=Lambda,
        cycle_Mu=cycle_Mu,
        cycle_Lambda=cycle_Lambda,
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
    init_nodes_pos, init_edges = (
        adata.uns[key]["node_pos"],
        adata.uns[key]["edge"],
    )
    PG = elpigraph.ExtendLeaves(
        adata.obsm["X_dr"].astype(float),
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
    _store_graph_attributes(adata, adata.obsm["X_dr"], key)


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
