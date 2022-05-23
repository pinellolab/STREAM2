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
                "use_partition should be a bool or a list of partitions"
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
                "adata.obs['pointweights'] not found. Please run"
                " st2.tl.get_weights"
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
    init_nodes_pos, init_edges = (
        adata.uns["epg"]["node_pos"],
        adata.uns["epg"]["edge"],
    )

    # --- Init parameters, variables
    if Mu is None:
        Mu = adata.uns[key]["params"]["epg_mu"]
    if Lambda is None:
        Lambda = adata.uns[key]["params"]["epg_lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu / 10
    if cycle_Lambda is None:
        cycle_Lambda = Lambda / 10
    if n_nodes is None:
        n_nodes = min(16, max(6, len(init_nodes_pos) / 20))
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, init_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )
    clus = (part == source) | (part == target)
    X_fit = np.vstack(
        (init_nodes_pos[source], init_nodes_pos[target], X[clus.flat])
    )

    # --- fit path
    _adata = sc.AnnData(X_fit)
    learn_graph(
        _adata,
        method="principal_curve",
        obsm=None,
        use_seed=False,
        epg_lambda=Lambda,
        epg_mu=Mu,
        n_nodes=n_nodes,
        FixNodesAtPoints=[[0], [1]],
    )

    # --- get nodep, edges, create new graph with added loop
    nodep, edges = _adata.uns["epg"]["node_pos"], _adata.uns["epg"]["edge"]

    _edges = edges.copy()
    _edges[(edges != 0) & (edges != 1)] += init_edges.max() - 1
    _edges[edges == 0] = source
    _edges[edges == 1] = target
    _merged_edges = np.concatenate((init_edges, _edges))
    _merged_nodep = np.concatenate((init_nodes_pos, nodep[2:]))

    if refit_graph:
        cycle_nodes = elpigraph._graph_editing.find_all_cycles(
            nx.Graph(_merged_edges.tolist())
        )[0]

        ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
            _merged_edges,
            Lambda=Lambda,
            Mu=Mu,
            cycle_Lambda=cycle_Lambda,
            cycle_Mu=cycle_Mu,
            cycle_nodes=cycle_nodes,
        )

        (
            _merged_nodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X,
            _merged_nodep,
            ElasticMatrix,
            PointWeights=weights,
            FixNodesAtPoints=[],
        )

    # check intersection
    if _merged_nodep.shape[1] == 2:
        intersect = not (
            MultiLineString(
                [LineString(_merged_nodep[e]) for e in _merged_edges]
            ).is_simple
        )
        if intersect:
            (
                _merged_nodep,
                _merged_edges,
            ) = elpigraph._graph_editing.remove_intersections(
                _merged_nodep, _merged_edges
            )

    adata.uns[key]["node_pos"] = _merged_nodep
    adata.uns[key]["edge"] = _merged_edges

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
        cycle_Mu = Mu / 10
    if cycle_Lambda is None:
        cycle_Lambda = Lambda / 10
    if use_weights:
        weights = np.array(adata.obs["pointweights"])[:, None]
    else:
        weights = None

    # --- get path to remove
    epg_edge = adata.uns[key]["edge"]
    epg_edge_len = adata.uns[key]["edge_len"]
    G = nx.Graph()
    G.add_nodes_from(range(adata.uns[key]["node_pos"].shape[0]))
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")

    if nodes_to_include is None:
        # nodes on the shortest path
        nodes_sp = nx.shortest_path(
            G, source=source, target=target, weight="len"
        )
    else:
        assert isinstance(
            nodes_to_include, list
        ), "`nodes_to_include` must be list"
        # lists of simple paths, in order from shortest to longest
        list_paths = list(
            nx.shortest_simple_paths(
                G, source=source, target=target, weight="len"
            )
        )
        flag_exist = False
        for p in list_paths:
            if set(nodes_to_include).issubset(p):
                nodes_sp = p
                flag_exist = True
                break
        if not flag_exist:
            print(f"no path that passes {nodes_to_include} exists")

    G.remove_edges_from(np.vstack((nodes_sp[:-1], nodes_sp[1:])).T)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    Gdel = nx.relabel_nodes(G, dict(zip(G.nodes, np.arange(len(G.nodes)))))

    adata.uns[key]["edge"] = np.array(Gdel.edges)
    adata.uns[key]["node_pos"] = adata.uns[key]["node_pos"][
        ~np.isin(range(len(adata.uns[key]["node_pos"])), isolates)
    ]

    # --- get nodep, edges, create new graph with added loop
    if refit_graph:
        nodep, edges = adata.uns["epg"]["node_pos"], adata.uns["epg"]["edge"]

        cycle_nodes = elpigraph._graph_editing.find_all_cycles(
            nx.Graph(edges.tolist())
        )[0]

        ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
            edges,
            Lambda=Lambda,
            Mu=Mu,
            cycle_Lambda=cycle_Lambda,
            cycle_Mu=cycle_Mu,
            cycle_nodes=cycle_nodes,
        )

        (
            newnodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X, nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[]
        )
        adata.uns["epg"]["node_pos"] = newnodep

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
    edges = np.argwhere(adata.uns["paga"]["connectivities"])
    edges_tree = np.argwhere(adata.uns["paga"]["connectivities_tree"])
    g = nx.convert_matrix.from_scipy_sparse_matrix(
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
