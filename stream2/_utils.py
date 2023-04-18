"""Utility functions and classes."""

from kneed import KneeLocator
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
import scipy


def locate_elbow(
    x,
    y,
    S=10,
    min_elbow=0,
    curve="convex",
    direction="decreasing",
    online=False,
    **kwargs,
):
    """Detect knee points
    Parameters
    ----------
    x : `array-like`
        x values
    y : `array-like`
        y values
    S : `float`, optional (default: 10)
        Sensitivity
    min_elbow: `int`, optional (default: 0)
        The minimum elbow location
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.
    Returns
    -------
    elbow: `int`
        elbow point
    """
    kneedle = KneeLocator(
        x[int(min_elbow) :],
        y[int(min_elbow) :],
        S=S,
        curve=curve,
        direction=direction,
        online=online,
        **kwargs,
    )
    if kneedle.elbow is None:
        elbow = len(y)
    else:
        elbow = int(kneedle.elbow)
    return elbow


def get_path(
    adata, source=None, target=None, nodes_to_include=None, key="epg"
):
    #### Extract cells by provided nodes

    epg_edge = adata.uns[key]["edge"]
    epg_edge_len = adata.uns[key]["edge_len"]
    G = nx.Graph()
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")

    if source is None:
        source = adata.uns[f"{key}_pseudotime_params"]["source"]
    if target is None:
        target = adata.uns[f"{key}_pseudotime_params"]["target"]
    if nodes_to_include is None:
        nodes_to_include = adata.uns[f"{key}_pseudotime_params"][
            "nodes_to_include"
        ]

    if target is not None:
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
    else:
        nodes_sp = [source] + [v for u, v in nx.bfs_edges(G, source)]

    cells = adata.obs_names[np.isin(adata.obs[f"{key}_node_id"], nodes_sp)]
    path_alias = "Path_%s-%s-%s" % (source, nodes_to_include, target)
    print(
        len(cells),
        "Cells are selected for Path_Source_Nodes-to-include_Target : ",
        path_alias,
    )
    return cells, path_alias


def get_expdata(
    adata, source=None, target=None, nodes_to_include=None, key="epg"
):
    cells, path_alias = get_path(adata, source, target, nodes_to_include, key)

    if scipy.sparse.issparse(adata.X):
        mat = adata.X.todense()
    else:
        mat = adata.X

    df_sc = pd.DataFrame(
        index=adata.obs_names.tolist(),
        data=mat,
        columns=adata.var.index.tolist(),
    )
    df_cells = deepcopy(df_sc.loc[cells])
    df_cells[f"{key}_pseudotime"] = adata.obs[f"{key}_pseudotime"][cells]
    df_cells_sort = df_cells.sort_values(
        by=[f"{key}_pseudotime"], ascending=True
    )

    return df_cells_sort, path_alias


def stream2elpi(adata, key="epg"):
    PG = {
        "NodePositions": adata.uns[key]["node_pos"].astype(float),
        "Edges": [
            adata.uns[key]["edge"],
            np.repeat(
                adata.uns[key]["params"]["epg_lambda"],
                len(adata.uns[key]["node_pos"]),
            ),
        ],
        "Lambda": adata.uns[key]["params"]["epg_lambda"],
        "Mu": adata.uns[key]["params"]["epg_mu"],
        "projection": {"edge_len": adata.uns[key]["edge_len"]},
    }
    return PG
