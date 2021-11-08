"""Pseudotime inference"""

import numpy as np
import networkx as nx

def infer_pseudotime(adata, source, target=None, nodes_to_include=None, key="epg"):
    """Infer pseudotime
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    """

    epg_edge = adata.uns[key]["edge"]
    epg_edge_len = adata.uns[key]["edge_len"]
    G = nx.Graph()
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")
    if target is not None:
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
                return f"no path that passes {nodes_to_include} exists"
    else:
        nodes_sp = [source] + [v for u, v in nx.bfs_edges(G, source)]
    G_sp = G.subgraph(nodes_sp).copy()
    index_nodes = {
        x: nodes_sp.index(x) if x in nodes_sp else G.number_of_nodes() for x in G.nodes
    }

    dict_dist_to_source = nx.shortest_path_length(G_sp, source=source, weight="len")

    cells = adata.obs_names[np.isin(adata.obs[f"{key}_node_id"], nodes_sp)]
    id_edges_cell = adata.obs.loc[cells, f"{key}_edge_id"].tolist()
    edges_cell = adata.uns[key]["edge"][id_edges_cell, :]
    len_edges_cell = adata.uns[key]["edge_len"][id_edges_cell]

    # proportion on the edge
    prop_edge = np.clip(
        adata.obs.loc[cells, f"{key}_edge_loc"], a_min=0, a_max=1
    ).values

    dist_to_source = []
    for i in np.arange(edges_cell.shape[0]):
        if index_nodes[edges_cell[i, 0]] > index_nodes[edges_cell[i, 1]]:
            dist_to_source.append(dict_dist_to_source[edges_cell[i, 1]])
            prop_edge[i] = 1 - prop_edge[i]
        else:
            dist_to_source.append(dict_dist_to_source[edges_cell[i, 0]])
    dist_to_source = np.array(dist_to_source)
    dist_on_edge = len_edges_cell * prop_edge
    dist = dist_to_source + dist_on_edge

    adata.obs[f"{key}_pseudotime"] = np.nan
    adata.obs.loc[cells, f"{key}_pseudotime"] = dist
    adata.uns[f'{key}_pseudotime_params'] = {'source':source,
                                             'target':target,
                                             'nodes_to_include':nodes_to_include}