"""Functions to calculate principal graph."""

import numpy as np
import pandas as pd
import scipy
import elpigraph
import networkx as nx
from copy import deepcopy
from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances

from .._settings import settings


def learn_graph(
        adata,
        method="principal_tree",
        obsm="X_dr",
        layer=None,
        n_nodes=50,
        epg_lambda=0.01,
        epg_mu=0.1,
        epg_alpha=0.02,
        use_seed=True,
        use_partition=False,
        use_weights=False,
        n_jobs=None,
        ordinal_labels=None,
        ordinal_supervision_strength=1,
        ordinal_root_point=None,
        **kwargs,
):
    """Learn principal graph.

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'principal_curve');
        Method used to calculate the graph.
    obsm: `str`, optional (default: 'X_dr')
        The multi-dimensional annotation of observations
        used to learn the graph
    layer: `str`, optional (default: None)
        The layer used to learn the graph
    use_seed: bool
        Whether to use the seed graph in adata.uns['seed_epg']
        generated by st.seed_graph.
        If True, ignores obsm and layer parameters
    **kwargs:
        Additional arguments to each method

    Returns
    -------
    updates `adata.uns['epg']` with the following field.
    conn: `sparse matrix` (`.uns['epg']['conn']`)
        A connectivity sparse matrix.
    node_pos: `array` (`.uns['epg']['node_pos']`)
        Node positions.
    """

    if use_partition:
        print("Learning elastic principal graph for each partition...")
        if type(use_partition) is bool:
            partitions = adata.obs["partition"].unique()
        elif type(use_partition) is list:
            partitions = use_partition
        else:
            raise ValueError(
                "use_partition should be a bool or a list of partitions"
            )
        if ordinal_labels is not None:
            raise ValueError(
                "use_partition can't be used together with ordinal_labels"
            )

        merged_nodep = []
        merged_edges = []
        num_edges = 0
        for part in adata.obs["partition"].unique():

            if part not in partitions:
                p_adata = _subset_adata(adata, part)
            else:
                if use_seed:
                    p_adata = _subset_adata(adata, part)
                    if len(p_adata.uns["seed_epg"]["node_pos"]) < n_nodes:
                        nnodes = n_nodes
                    else:
                        nnodes = len(p_adata.uns["seed_epg"]["node_pos"]) + 1
                else:
                    p_adata = adata[adata.obs["partition"] == part].copy()

                _learn_graph(
                    p_adata,
                    method=method,
                    obsm=obsm,
                    layer=layer,
                    n_nodes=nnodes,
                    epg_lambda=epg_lambda,
                    epg_mu=epg_mu,
                    epg_alpha=epg_alpha,
                    use_seed=use_seed,
                    use_weights=use_weights,
                    n_jobs=n_jobs,
                    **kwargs,
                )

            merged_nodep.append(p_adata.uns["epg"]["node_pos"])
            merged_edges.append(p_adata.uns["epg"]["edge"] + num_edges)
            num_edges += len(p_adata.uns["epg"]["node_pos"])

        adata.uns["epg"] = {}
        adata.uns["epg"]["node_pos"] = np.concatenate(merged_nodep)
        adata.uns["epg"]["edge"] = np.concatenate((merged_edges))
        adata.uns["epg"]["node_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(nodep) for nodep in merged_nodep],
        ).astype(str)
        adata.uns["epg"]["edge_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(edges) for edges in merged_edges],
        ).astype(str)
        adata.uns["epg"]["params"] = p_adata.uns["epg"]["params"]

        X = _get_graph_data(adata, key="epg")
        _store_graph_attributes(adata, X, key="epg")

    else:
        _learn_graph(
            adata,
            method=method,
            obsm=obsm,
            layer=layer,
            n_nodes=n_nodes,
            epg_lambda=epg_lambda,
            epg_mu=epg_mu,
            epg_alpha=epg_alpha,
            use_seed=use_seed,
            use_weights=use_weights,
            ordinal_labels=ordinal_labels,
            ordinal_supervision_strength=ordinal_supervision_strength,
            ordinal_root_point=ordinal_root_point,
            n_jobs=n_jobs,
            **kwargs,
        )


def _learn_graph(
        adata,
        method="principal_tree",
        obsm="X_dr",
        layer=None,
        n_nodes=50,
        epg_lambda=0.01,
        epg_mu=0.1,
        epg_alpha=0.02,
        use_seed=True,
        use_weights=False,
        ordinal_labels=None,
        ordinal_supervision_strength=1,
        ordinal_root_point=None,
        n_jobs=None,
        **kwargs,
):
    """Learn principal graph.

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'principal_curve');
        Method used to calculate the graph.
    obsm: `str`, optional (default: 'X_dr')
        The multi-dimensional annotation of observations
        used to learn the graph
    layer: `str`, optional (default: None)
        The layer used to learn the graph
    use_seed: bool
        Whether to use the seed graph in adata.uns['seed_epg']
        generated by st.seed_graph.
        If True, ignores obsm and layer parameters
    **kwargs:
        Additional arguments to each method

    Returns
    -------
    updates `adata.uns['epg']` with the following field.
    conn: `sparse matrix` (`.uns['epg']['conn']`)
        A connectivity sparse matrix.
    node_pos: `array` (`.uns['epg']['node_pos']`)
        Node positions.
    """

    assert method in [
        "principal_curve",
        "principal_tree",
        "principal_circle",
    ], (
        "`method` must be one of "
        "['principal_curve','principal_tree','principal_circle']"
    )

    if use_seed and (method == "principal_tree"):
        if "seed_epg" not in adata.uns:
            raise ValueError(
                "could not find 'seed_epg' in `adata.uns. Please run"
                " st.tl.seed_graph"
            )
        if n_nodes <= len(adata.uns["seed_epg"]["node_pos"]):
            raise ValueError(
                f"The seed graph already has at least {n_nodes} nodes. Please"
                " run st.tl.learn_graph with higher n_nodes"
            )
        kwargs["InitNodePositions"] = adata.uns["seed_epg"]["node_pos"]
        kwargs["InitEdges"] = adata.uns["seed_epg"]["edge"]
        if adata.uns["seed_epg"]["params"]["obsm"] is not None:
            mat = adata.obsm[adata.uns["seed_epg"]["params"]["obsm"]]
        elif adata.uns["seed_epg"]["params"]["layer"] is not None:
            mat = adata.obsm[adata.uns["seed_epg"]["params"]["layer"]]
        else:
            print("Learning the graph on adata.X")
            mat = adata.X
    else:
        if (
                use_seed
                and (method != "principal_tree")
                and ("seed_epg" in adata.uns)
        ):
            print(
                f"WARNING: seed graph is ignored when using method {method}"
            )

        kwargs["InitNodePositions"] = None
        kwargs["InitEdges"] = None
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

    if n_jobs is None:
        n_jobs = settings.n_jobs

    if "MaxNumberOfGraphCandidatesDict" not in kwargs:
        kwargs["MaxNumberOfGraphCandidatesDict"] = {
            "AddNode2Node": 20,
            "BisectEdge": 20,
            "ShrinkEdge": 50,
        }
    if use_weights:
        if "pointweights" not in adata.obs:
            raise ValueError(
                "adata.obs['pointweights'] not found. Please run"
                " st2.tl.get_weights"
            )
        weights = np.array(adata.obs["pointweights"]).reshape((-1, 1))
    else:
        weights = None

    if ordinal_labels is not None:
        kwargs["pseudotime"] = ordinal_labels
        kwargs["pseudotimeLambda"] = ordinal_supervision_strength
        kwargs["FixNodesAtPoints"] = [ordinal_root_point]

    if method == "principal_curve":
        dict_epg = elpigraph.computeElasticPrincipalCurve(
            X=mat,
            NumNodes=n_nodes,
            n_cores=n_jobs,
            Do_PCA=False,
            CenterData=False,
            Lambda=epg_lambda,
            Mu=epg_mu,
            alpha=epg_alpha,
            PointWeights=weights,
            **kwargs,
        )[0]
    if method == "principal_tree":
        dict_epg = elpigraph.computeElasticPrincipalTree(
            X=mat,
            NumNodes=n_nodes,
            n_cores=n_jobs,
            Do_PCA=False,
            CenterData=False,
            Lambda=epg_lambda,
            Mu=epg_mu,
            alpha=epg_alpha,
            PointWeights=weights,
            **kwargs,
        )[0]
    if method == "principal_circle":
        dict_epg = elpigraph.computeElasticPrincipalCircle(
            X=mat,
            NumNodes=n_nodes,
            n_cores=n_jobs,
            Do_PCA=False,
            CenterData=False,
            Lambda=epg_lambda,
            Mu=epg_mu,
            alpha=epg_alpha,
            PointWeights=weights,
            InitNodes=3,
            **kwargs,
        )[0]

    adata.uns["epg"] = dict()

    adata.uns["epg"]["node_pos"] = dict_epg["NodePositions"]
    adata.uns["epg"]["edge"] = dict_epg["Edges"][0]
    adata.uns["epg"]["params"] = {
        "method": method,
        "obsm": obsm,
        "layer": layer,
        "n_nodes": n_nodes,
        "epg_lambda": epg_lambda,
        "epg_mu": epg_mu,
        "epg_alpha": epg_alpha,
        "use_seed": use_seed,
    }
    if "StoreGraphEvolution" in kwargs:
        adata.uns["epg"]["graph_evolution"] = {
            "all_node_pos": dict_epg["AllNodePositions"],
            "all_edge": dict_epg["AllElasticMatrices"],
        }
    _store_graph_attributes(adata, mat, key="epg")


def seed_graph(
        adata,
        obsm="X_dr",
        layer=None,
        clustering="kmeans",
        damping=0.75,
        pref_perc=50,
        n_clusters=10,
        max_n_clusters=200,
        n_neighbors=50,
        nb_pct=None,
        paths=[],
        paths_forbidden=[],
        label=None,
        label_strength=0.5,
        force=False,
        use_weights=False,
        use_partition=False,
):
    """Seeding the initial elastic principal graph.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    obsm: `str`, optional (default: 'X_dr')
        The multi-dimensional annotation of observations
        used to learn the graph
    layer: `str`, optional (default: None)
        The layer used to learn the graph
    init_nodes_pos: `array`, shape = [n_nodes,n_dimension],
        optional (default: `None`)
        initial node positions
    init_edges: `array`, shape = [n_edges,2], optional (default: `None`)
        initial edges, all the initial nodes should be included
        in the tree structure
    clustering: `str`, optional (default: 'kmeans')
        Choose from {{'ap','kmeans','sc'}}
        clustering method used to infer the initial nodes.
        'ap' affinity propagation
        'kmeans' K-Means clustering
        'sc' spectral clustering
    damping: `float`, optional (default: 0.75)
        Damping factor (between 0.5 and 1) for affinity propagation.
    pref_perc: `int`, optional (default: 50)
        Preference percentile (between 0 and 100).
        The percentile of the input similarities for affinity propagation.
    n_clusters: `int`, optional (default: 10)
        Number of clusters (only valid once 'clustering'
        is specified as 'sc' or 'kmeans').
    max_n_clusters: `int`, optional (default: 200)
        The allowed maximum number of clusters for 'ap'.
    n_neighbors: `int`, optional (default: 50)
        The number of neighbor cells used for spectral clustering.
    nb_pct: `float`, optional (default: None)
        The percentage of neighbor cells
        (when specified, it will overwrite n_neighbors).
    use_vis: `bool`, optional (default: False)
        If True, the manifold learnt from st.plot_visualization_2D()
        will replace the manifold learnt from st.dimension_reduction().
        The principal graph will be learnt in the new manifold.
    paths: list of lists, optional (default: [])
        Paths between categorical labels used
        for supervised MST initialization
    paths_forbidden: list of lists, optional (default: [])
        Forbidden paths between categorical labels
        used for supervised MST initialization
    labels: `str`, optional (default: None)
        Categorical labels for supervised MST initialization

    Returns
    -------
    updates `adata` with the following fields.
    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns
        of current root_node_pseudotime and removing the previous ones.
    clustering: `pandas.core.series.Series`
        (`adata.obs['clustering']`,dtype `str`)
        Array of dim (number of samples) that stores
        the clustering labels ('0', '1', …) for each cell.
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure.
        It contains node attributes ('pos')
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure
        by only keeping leaf nodes and branching nodes.
        It contains node attributes ('pos','label')
        and edge attributes ('nodes','id','len','color').
    seed_epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Store seeded elastic principal graph structure
    seed_flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        Store seeded flat_tree
    Notes
    -------
    The default procedure is fast and good enough
    when seeding structure in low-dimensional space.
    when seeding structure in high-dimensional space,
    it's strongly recommended to use 'infer_initial_structure'
     to get the initial node positions and edges
    """

    if use_partition:
        print("Seeding initial graph for each partition...")
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
            if type(use_partition) is list:
                p_adata = _subset_adata(adata, part)
            else:
                p_adata = adata[adata.obs["partition"] == part].copy()

            if part in partitions:
                _seed_graph(
                    p_adata,
                    obsm=obsm,
                    layer=layer,
                    clustering=clustering,
                    damping=damping,
                    pref_perc=pref_perc,
                    n_clusters=n_clusters,
                    max_n_clusters=max_n_clusters,
                    n_neighbors=n_neighbors,
                    nb_pct=nb_pct,
                    paths=paths,
                    paths_forbidden=paths_forbidden,
                    label=label,
                    label_strength=label_strength,
                    force=force,
                    use_weights=use_weights,
                    verbose=False,
                )

            merged_nodep.append(p_adata.uns["seed_epg"]["node_pos"])
            merged_edges.append(p_adata.uns["seed_epg"]["edge"] + num_edges)
            num_edges += len(p_adata.uns["seed_epg"]["node_pos"])

        adata.uns["seed_epg"] = {}
        adata.uns["seed_epg"]["node_pos"] = np.concatenate(merged_nodep)
        adata.uns["seed_epg"]["edge"] = np.concatenate((merged_edges))
        adata.uns["seed_epg"]["node_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(nodep) for nodep in merged_nodep],
        ).astype(str)
        adata.uns["seed_epg"]["edge_partition"] = np.repeat(
            adata.obs["partition"].unique(),
            [len(edges) for edges in merged_edges],
        ).astype(str)
        adata.uns["seed_epg"]["params"] = p_adata.uns["seed_epg"]["params"]

        X = _get_graph_data(adata, key="seed_epg")
        _store_graph_attributes(adata, X, key="seed_epg")

    else:
        _seed_graph(
            adata,
            obsm=obsm,
            layer=layer,
            clustering=clustering,
            damping=damping,
            pref_perc=pref_perc,
            n_clusters=n_clusters,
            max_n_clusters=max_n_clusters,
            n_neighbors=n_neighbors,
            nb_pct=nb_pct,
            paths=paths,
            paths_forbidden=paths_forbidden,
            label=label,
            label_strength=label_strength,
            force=force,
            use_weights=use_weights,
        )


def _seed_graph(
        adata,
        obsm="X_dr",
        layer=None,
        clustering="kmeans",
        damping=0.75,
        pref_perc=50,
        n_clusters=10,
        max_n_clusters=200,
        n_neighbors=50,
        nb_pct=None,
        paths=[],
        paths_forbidden=[],
        label=None,
        label_strength=0.5,
        force=False,
        use_weights=False,
        verbose=True,
):
    """Internal method to seed_graph"""

    if verbose:
        print("Seeding initial graph...")

    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm:
            mat = adata.obsm[obsm]
            adata.uns["seed"] = obsm
        else:
            raise ValueError(f"could not find {obsm} in `adata.obsm`")
    elif layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer]
            adata.uns["seed"] = obsm
        else:
            raise ValueError(f"could not find {layer} in `adata.layers`")
    else:
        mat = adata.X

    if nb_pct is not None:
        n_neighbors = int(np.around(mat.shape[0] * nb_pct))

    if verbose:
        print("Clustering...")
    if clustering == "ap":
        if verbose:
            print("Affinity propagation ...")
        ap = AffinityPropagation(
            damping=damping,
            random_state=42,
            preference=np.percentile(
                -euclidean_distances(mat, squared=True), pref_perc
            ),
        ).fit(mat)
        # ap = AffinityPropagation(damping=damping).fit(mat)
        if ap.cluster_centers_.shape[0] > max_n_clusters:
            if verbose:
                print(
                    "The number of clusters is "
                    + str(ap.cluster_centers_.shape[0])
                )
            if verbose:
                print(
                    "Too many clusters are generated, please lower pref_perc"
                    " or increase damping and retry it"
                )
            return
        cluster_labels = ap.labels_
        init_nodes_pos = ap.cluster_centers_
    elif clustering == "sc":
        if verbose:
            print("Spectral clustering ...")
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            eigen_solver="arpack",
            random_state=42,
        ).fit(mat)
        cluster_labels = sc.labels_
        init_nodes_pos = np.empty((0, mat.shape[1]))  # cluster centers
        for x in np.unique(cluster_labels):
            id_cells = np.array(range(mat.shape[0]))[cluster_labels == x]
            init_nodes_pos = np.vstack(
                (init_nodes_pos, np.median(mat[id_cells, :], axis=0))
            )
    elif clustering == "kmeans":
        if verbose:
            print("K-Means clustering ...")
        if use_weights:
            if "pointweights" not in adata.obs:
                raise ValueError(
                    "adata.obs['pointweights'] not found. Please run"
                    " st2.tl.get_weights"
                )
        else:
            weights = None
        kmeans = KMeans(
            n_clusters=n_clusters, init="k-means++", random_state=42
        ).fit(mat, sample_weight=weights)
        cluster_labels = kmeans.labels_
        init_nodes_pos = kmeans.cluster_centers_
    else:
        if verbose:
            print("'" + clustering + "'" + " is not supported")
    adata.obs[clustering] = ["cluster " + str(x) for x in cluster_labels]

    # Minimum Spanning Tree ###
    if verbose:
        print("Calculating minimum spanning tree...")

    # ---if supervised adjacency matrix option
    if (
            ((len(paths) > 0) or (len(paths_forbidden) > 0)) and label is None
    ) or (
            ((len(paths) == 0) and (len(paths_forbidden) == 0))
            and label is not None
    ):
        raise ValueError(
            "Both a label key (label: str) and cluster paths (paths: list of"
            " list) need to be provided for path-supervised initialization"
        )
    elif (
            (len(paths) > 0) or (len(paths_forbidden) > 0)
    ) and label is not None:
        (
            init_nodes_pos,
            clus_adjmat,
            adjmat,
            adjmat_strength,
            num_modes,
            num_labels,
            labels_ignored,
        ) = _categorical_adjmat2(
            mat,
            init_nodes_pos,
            paths,
            paths_forbidden,
            adata.obs[label],
            label_strength,
        )
        D = pairwise_distances(init_nodes_pos)
        G = nx.from_numpy_array(D * clus_adjmat)

    # ---else unsupervised
    else:
        D = pairwise_distances(init_nodes_pos)
        G = nx.from_numpy_array(D)

    # ---get edges from mst
    mst = nx.minimum_spanning_tree(G, ignore_nan=True)
    init_edges = np.array(mst.edges())
    if force and label is not None:
        init_edges = _force_missing_connections(
            D, num_labels, num_modes, init_edges, paths, clus_adjmat
        )

    # Store results ###
    adata.uns["seed_epg"] = dict()
    adata.uns["seed_epg"]["node_pos"] = init_nodes_pos
    adata.uns["seed_epg"]["edge"] = init_edges
    adata.uns["seed_epg"]["params"] = dict(
        obsm=obsm,
        layer=layer,
        clustering=clustering,
        damping=damping,
        pref_perc=pref_perc,
        n_clusters=n_clusters,
        max_n_clusters=max_n_clusters,
        n_neighbors=n_neighbors,
        nb_pct=nb_pct,
    )
    _store_graph_attributes(adata, mat, key="seed_epg")


def _store_graph_attributes(adata, mat, key):
    """Compute graph attributes and store them in adata.uns[key]"""

    G = nx.Graph()
    G.add_edges_from(adata.uns[key]["edge"].tolist(), weight=1)
    mat_conn = nx.to_scipy_sparse_matrix(
        G,
        nodelist=np.arange(len(adata.uns[key]["node_pos"])),
        weight="weight",
    )

    # partition points
    node_id, node_dist = elpigraph.src.core.PartitionData(
        X=mat,
        NodePositions=adata.uns[key]["node_pos"],
        MaxBlockSize=len(adata.uns[key]["node_pos"]) ** 4,
        SquaredX=np.sum(mat ** 2, axis=1, keepdims=1),
    )
    # project points onto edges
    dict_proj = elpigraph.src.reporting.project_point_onto_graph(
        X=mat,
        NodePositions=adata.uns[key]["node_pos"],
        Edges=adata.uns[key]["edge"],
        Partition=node_id,
    )

    adata.obs[f"{key}_node_id"] = node_id.flatten()
    adata.obs[f"{key}_node_dist"] = node_dist
    adata.obs[f"{key}_edge_id"] = dict_proj["EdgeID"].astype(int)
    adata.obs[f"{key}_edge_loc"] = dict_proj["ProjectionValues"]

    # adata.obsm[f"X_{key}_proj"] = dict_proj["X_projected"]

    adata.uns[key]["conn"] = mat_conn
    adata.uns[key]["edge_len"] = dict_proj["EdgeLen"]


def _get_branch_id(adata, key="epg"):
    """add adata.obs['branch_id']"""
    # get branches
    net = elpigraph.src.graphs.ConstructGraph(
        {"Edges": [adata.uns[key]["edge"]]}
    )
    branches = elpigraph.src.graphs.GetSubGraph(net, "branches")
    _dict_branches = {
        (b[0], b[-1]): b for i, b in enumerate(branches)
    }  # temporary branch node lists (not in order)

    ordered_edges, ordered_nodes = elpigraph.src.supervised.bf_search(
        _dict_branches, root_node=np.where(np.array(net.degree()) == 1)[0][0]
    )
    # create ordered dict
    dict_branches = {}
    for i, e in enumerate(ordered_edges):  # for each branch
        # store branch in order (both the key and the list)
        if e not in _dict_branches:
            dict_branches[e] = _dict_branches[e[::-1]][::-1]
        else:
            dict_branches[e] = _dict_branches[e]

    # disable warning
    pd.options.mode.chained_assignment = None

    point_edges = adata.uns[key]["edge"][adata.obs[f"{key}_edge_id"]]
    adata.obs[f"{key}_branch_id"] = ""
    for i, e in enumerate(point_edges):
        for k, v in dict_branches.items():
            if all(np.isin(e, v)):
                adata.obs[f"{key}_branch_id"][i] = k

    # reactivate warning
    pd.options.mode.chained_assignment = "warn"


# Categorical MST initialization utils ##

def _force_missing_connections(
        D, num_labels, num_modes, init_edges, paths, clus_adjmat
):
    found_missing = True
    while found_missing:

        found_missing = False
        edges_labels = np.array(list(num_labels.keys()))[num_modes][
            init_edges
        ].tolist()
        for path in paths:
            for i in range(len(path) - 1):
                if [path[i], path[i + 1]] not in edges_labels and [
                    path[i + 1],
                    path[i],
                ] not in edges_labels:
                    print(path[i], path[i + 1])
                    print(num_labels[path[i]], num_labels[path[i + 1]])

                    missing_is = np.where(num_modes == num_labels[path[i]])[0]
                    missing_js = np.where(
                        num_modes == num_labels[path[i + 1]]
                    )[0]
                    x = D[missing_is[:, None], missing_js]
                    _i, _j = np.where(x == x.min())
                    mi, mj = missing_is[_i], missing_js[_j]
                    D[mi, mj] = D[mj, mi] = -1.0
                    found_missing = True

        # ---get edges from mst
        G = nx.from_numpy_array(D * clus_adjmat)
        mst = nx.minimum_spanning_tree(G, ignore_nan=True)
        init_edges = np.array(mst.edges())
    return init_edges


def _get_partition_modes(mat, init_nodes_pos, labels):
    """Return most frequent label assigned to each node."""
    labels = np.array(labels)
    part = elpigraph.src.core.PartitionData(
        mat, init_nodes_pos, 10 ** 6, np.sum(mat ** 2, axis=1, keepdims=1)
    )[0].flatten()
    modes = np.empty(len(init_nodes_pos), dtype=labels.dtype)

    for i in range(len(init_nodes_pos)):
        modes[i] = scipy.stats.mode(labels[part == i]).mode[0]
    return modes


def _get_labels_adjmat(labels_u, labels_ignored, paths, paths_forbidden):
    """Create adjmat given labels and paths.

    labels_ignored are connected to all other labels
    """
    num_labels = {
        s: i for i, s in enumerate(np.append(labels_u, labels_ignored))
    }
    adjmat = np.zeros(
        (
            len(labels_u) + len(labels_ignored),
            len(labels_u) + len(labels_ignored),
        ),
        dtype=int,
    )

    # allow within-cluster connections
    np.fill_diagonal(adjmat, 1)

    # allow connections given from paths
    for p in paths:
        for i in range(len(p) - 1):
            adjmat[num_labels[p[i]], num_labels[p[i + 1]]] = adjmat[
                num_labels[p[i + 1]], num_labels[p[i]]
            ] = 1

    # allow unspecified clusters to connect to all other clusters
    for l in labels_ignored:
        adjmat[num_labels[l]] = adjmat[:, num_labels[l]] = 1

    # remove forbidden connections given from paths_forbidden
    for p in paths_forbidden:
        for i in range(len(p) - 1):
            adjmat[num_labels[p[i]], num_labels[p[i + 1]]] = adjmat[
                num_labels[p[i + 1]], num_labels[p[i]]
            ] = 0

    return adjmat, num_labels


def _get_clus_adjmat(adjmat, num_modes, n_clusters):
    """Create clus_adjmat given labels adjmat and kmeans label assignment."""

    adjmat_clus = np.full((n_clusters, n_clusters), np.nan)
    eis, ejs = adjmat.nonzero()

    for ei, ej in zip(eis, ejs):
        clus_ei = np.where(num_modes == ei)[0]
        clus_ej = np.where(num_modes == ej)[0]
        adjmat_clus[
            clus_ei[:, None], np.repeat(clus_ej[None], len(clus_ei), axis=0)
        ] = 1
    return adjmat_clus


def _categorical_adjmat(mat, init_nodes_pos, paths, paths_forbidden, labels):
    """Main function, create categorical adjmat given node positions, cluster
    paths, point labels."""

    labels_u = np.unique([c for p in paths for c in p])
    labels_ignored = np.setdiff1d(labels, labels_u)
    # label adjacency matrix
    adjmat, num_labels = _get_labels_adjmat(
        labels_u, labels_ignored, paths, paths_forbidden
    )
    # assign label to nodes
    modes = _get_partition_modes(mat, init_nodes_pos, labels)
    num_modes = np.array([num_labels[m] for m in modes])

    # add centroids if necessary to prevent bug
    # (if some label has no kmean assigned to it)
    labels_miss = np.setdiff1d(labels_u, modes)
    if len(labels_miss) > 0:
        print(
            f"Found label(s) {labels_miss} with no representative node. Adding"
            " label centroid(s) as node(s)"
        )
        centroids = np.vstack(
            [mat[labels == s].mean(axis=0) for s in labels_miss]
        )
        init_nodes_pos = np.vstack((init_nodes_pos, centroids))
        modes = np.hstack((modes, labels_miss))
        num_modes = np.array([num_labels[m] for m in modes])

    # nodes adjacency matrix
    clus_adjmat = _get_clus_adjmat(
        adjmat, num_modes, n_clusters=len(init_nodes_pos)
    )
    return init_nodes_pos, clus_adjmat, adjmat, num_modes, num_labels


def _get_labels_adjmat2(labels_u, labels_ignored, paths, paths_forbidden):
    """Create adjmat given labels and paths.

    labels_ignored are connected to all other labels
    """
    num_labels = {
        s: i for i, s in enumerate(np.append(labels_u, labels_ignored))
    }
    len_labels = len(labels_u) + len(labels_ignored)
    adjmat = np.zeros((len_labels, len_labels))

    # allow within-cluster connections
    np.fill_diagonal(adjmat, 1)

    # allow connections given from paths
    for p in paths:
        for i in range(len(p) - 1):
            adjmat[num_labels[p[i]], num_labels[p[i + 1]]] = adjmat[
                num_labels[p[i + 1]], num_labels[p[i]]
            ] = 1

    # disallow unspecified clusters to connect to all other clusters
    for l in labels_ignored:
        adjmat[num_labels[l]] = adjmat[:, num_labels[l]] = 0
        adjmat[num_labels[l], num_labels[l]] = 1

    # remove forbidden connections given from paths_forbidden
    for p in paths_forbidden:
        for i in range(len(p) - 1):
            adjmat[num_labels[p[i]], num_labels[p[i + 1]]] = adjmat[
                num_labels[p[i + 1]], num_labels[p[i]]
            ] = np.nan

    return adjmat, num_labels


def _get_clus_adjmat2(adjmat_strength, num_modes, n_clusters, factor):
    """Create clus_adjmat given labels adjmat and kmeans label assignment."""

    adjmat_clus = np.ones((n_clusters, n_clusters))

    for ei in range(len(adjmat_strength)):
        for ej in range(len(adjmat_strength)):
            clus_ei = np.where(num_modes == ei)[0]
            clus_ej = np.where(num_modes == ej)[0]
            adjmat_clus[
                clus_ei[:, None],
                np.repeat(clus_ej[None], len(clus_ei), axis=0),
            ] = (1 - factor) + adjmat_strength[ei, ej] * factor
    return adjmat_clus


def _categorical_adjmat2(
        mat, init_nodes_pos, paths, paths_forbidden, labels, factor
):
    """Main function, create categorical adjmat given
    node positions, cluster paths, point labels."""

    labels_u = np.unique([c for p in paths for c in p])
    labels_ignored = np.setdiff1d(labels, labels_u)
    # label adjacency matrix
    adjmat, num_labels = _get_labels_adjmat2(
        labels_u, labels_ignored, paths, paths_forbidden
    )

    ix_nan = np.isnan(adjmat)
    adjmat[ix_nan] = 0.0
    graph = nx.from_numpy_array(adjmat)
    adjmat_strength = np.array(
        pd.DataFrame(dict(nx.all_pairs_shortest_path_length(graph)))
    )
    min_shortestpath = 1.0
    np.fill_diagonal(adjmat_strength, min_shortestpath)
    max_shortestpath = np.nanmax(adjmat_strength)
    adjmat_strength = adjmat_strength / max_shortestpath
    adjmat_strength[np.isnan(adjmat_strength) & (~ix_nan)] = 1.0

    # assign label to nodes
    modes = _get_partition_modes(mat, init_nodes_pos, labels)
    num_modes = np.array([num_labels[m] for m in modes])

    # add centroids if necessary to prevent bug
    # (if some label has no kmean assigned to it)
    labels_miss = np.setdiff1d(labels_u, modes)
    if len(labels_miss) > 0:
        print(
            f"Found label(s) {labels_miss} with no representative node. Adding"
            " label centroid(s) as node(s)"
        )
        centroids = np.vstack(
            [mat[labels == s].mean(axis=0) for s in labels_miss]
        )
        init_nodes_pos = np.vstack((init_nodes_pos, centroids))
        modes = np.hstack((modes, labels_miss))
        num_modes = np.array([num_labels[m] for m in modes])

    # nodes adjacency matrix
    clus_adjmat = _get_clus_adjmat2(
        adjmat_strength,
        num_modes,
        n_clusters=len(init_nodes_pos),
        factor=factor,
    )
    return (
        init_nodes_pos,
        clus_adjmat,
        adjmat,
        adjmat_strength,
        num_modes,
        num_labels,
        labels_ignored,
    )


def _get_graph_data(adata, key):
    """get data matrix used to learn the graph."""
    obsm = adata.uns[key]["params"]["obsm"]
    layer = adata.uns[key]["params"]["layer"]

    if obsm is not None:
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
    return mat


def _subset_adata(adata, part):
    p_adata = adata[adata.obs["partition"] == part].copy()
    for key in ["seed_epg", "epg"]:
        if key in p_adata.uns:
            p_adata.uns[key] = deepcopy(adata.uns[key])
            p_adata.uns[key]["node_pos"] = p_adata.uns[key]["node_pos"][
                p_adata.uns[key]["node_partition"] == part
                ]
            p_adata.uns[key]["edge"] = p_adata.uns[key]["edge"][
                p_adata.uns[key]["edge_partition"] == part
                ]
            p_adata.uns[key]["edge"] -= p_adata.uns[key]["edge"].min()
    return p_adata
