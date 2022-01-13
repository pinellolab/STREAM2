import networkx as nx
import elpigraph
import numpy as np
import scanpy as sc

from . import _graph_loops
from ._elpigraph import learn_graph, _store_graph_attributes


def _get_graph_data(adata, key):
    """get data matrix used to learn the graph"""
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


def add_path(adata, source, target, n_nodes=None, weights=None, key="epg"):

    X = _get_graph_data(adata, key)
    init_nodes_pos, init_edges = adata.uns["epg"]["node_pos"], adata.uns["epg"]["edge"]

    # --- Init parameters, variables
    Mu = adata.uns[key]["params"]["epg_mu"]
    Lambda = adata.uns[key]["params"]["epg_lamba"]
    if n_nodes is None:
        n_nodes = min(20, max(8, len(init_nodes_pos) // 6))
    if weights is None:
        weights = np.ones(len(X))[:, None]

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, init_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )
    clus = (part == source) | (part == target)
    X_fit = np.vstack((init_nodes_pos[source], init_nodes_pos[target], X[clus.flat]))

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
    cycle_edges = _graph_loops.find_all_cycles(nx.Graph(_merged_edges.tolist()))[0]

    Mus = np.repeat(Mu, len(_merged_nodep))
    Mus[cycle_edges] = Mu / 10000
    ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(
        _merged_edges, Lambdas=Lambda, Mus=Mus
    )
    _merged_nodep, _, _, _, _, _, _ = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
        X, _merged_nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[]
    )

    # check intersection
    if _merged_nodep.shape[1] == 2:
        intersect = not (
            _graph_loops.MultiLineString(
                [_graph_loops.LineString(_merged_nodep[e]) for e in _merged_edges]
            ).is_simple
        )
        if intersect:
            _merged_nodep, _merged_edges = _graph_loops.remove_intersections(
                _merged_nodep, _merged_edges
            )

    adata.uns[key]["node_pos"] = _merged_nodep
    adata.uns[key]["edge"] = _merged_edges

    # update edge_len, conn, data projection
    _store_graph_attributes(adata, X, key)


def del_path(adata, source, target, nodes_to_include=None, key="epg"):

    # --- get path to remove
    epg_edge = adata.uns[key]["edge"]
    epg_edge_len = adata.uns[key]["edge_len"]
    G = nx.Graph()
    G.add_nodes_from(range(adata.uns[key]["node_pos"].shape[0]))
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")

    if nodes_to_include is None:
        # nodes on the shortest path
        nodes_sp = nx.shortest_path(G, source=source, target=target, weight="len")
    else:
        assert isinstance(nodes_to_include, list), "`nodes_to_include` must be list"
        # lists of simple paths, in order from shortest to longest
        list_paths = list(
            nx.shortest_simple_paths(G, source=source, target=target, weight="len")
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

    # update edge_len, conn, data projection
    mat = _get_graph_data(adata, key)
    _store_graph_attributes(adata, mat, key)

