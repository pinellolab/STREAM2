"""Utility functions and classes."""

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
import itertools
from ..tools._pseudotime import infer_pseudotime


def split_at_values(lst, values):
    """Split a list of numbers based on given values."""

    list_split = list()
    indices = [i for i, x in enumerate(lst) if x in values]
    indices = [0, len(lst)-1] + indices
    indices = list(np.unique(indices))
    for start, end in list(zip(indices[:-1], indices[1:])):
        list_split.append(lst[start:end+1])
    return list_split


def _construct_stream_tree(adata, source=0, key="epg",):
    """construct a tree for stream plots"""
    if source not in adata.uns['epg']['node']:
        raise ValueError(f"There is no source {source}")
    epg_edge = adata.uns[key]["edge"]
    epg_edge_len = adata.uns[key]["edge_len"]
    G = nx.Graph()
    edges_weighted = list(zip(
        epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")

    # get all branches for stream plots
    target_nodes = [
        n for n, d in G.degree() if d == 1 or d > 2]
    all_paths = list(
        nx.all_simple_paths(
            G,
            source=source,
            target=list(set(target_nodes) - set([source]))))

    dict_branches = dict()
    for i, p in enumerate(all_paths):
        if len(set(p).intersection(
                set([source] + target_nodes))) == 2:
            dict_branches[(p[0], p[-1])] = p
        else:
            branches_i = split_at_values(p, [source] + target_nodes)
            for pp in branches_i:
                if pp not in dict_branches.values():
                    dict_branches[(pp[0], pp[-1])] = pp

    list_edge = list()
    list_edge_len = list()
    list_edge_nodes = list()
    for x in dict_branches.keys():
        list_edge.append(list(x))
        list_edge_len.append(
            nx.shortest_path_length(G, source=x[0], target=x[1], weight='len'))
        list_edge_nodes.append(dict_branches[x])

    adata.uns['stream_tree'] = dict()
    adata.uns['stream_tree']['params'] = dict(
        source=source,
        key=key)
    adata.uns['stream_tree']['node'] = \
        np.unique(list(itertools.chain(*list_edge)))
    adata.uns['stream_tree']['edge'] = np.array(list_edge)
    adata.uns['stream_tree']['edge_len'] = np.array(list_edge_len)
    adata.uns['stream_tree']['edge_nodes'] = pd.Series(list_edge_nodes)


# modified depth first search
def _dfs_nodes_modified(tree, source, preference=None):

    visited, stack = [], [source]
    bfs_tree = nx.bfs_tree(tree, source=source)
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            unvisited = set(tree[vertex]) - set(visited)
            if preference is not None:
                weights = list()
                for x in unvisited:
                    successors = dict(
                        nx.bfs_successors(bfs_tree, source=x))
                    successors_nodes = list(
                        itertools.chain.from_iterable(successors.values()))
                    weights.append(
                        min([preference.index(s)
                             if s in preference else len(preference)
                             for s in successors_nodes+[x]]))
                unvisited = [x for _, x in sorted(
                    zip(weights, unvisited), reverse=True, key=lambda x: x[0])]
            stack.extend(unvisited)
    return visited


def _calculate_shift_distance(
        adata,
        source=0,
        dist_pctl=95,
        preference=None
):
    """calculate shifting distance for each branch."""
    stream_tree = nx.Graph()
    stream_tree_edge = adata.uns['stream_tree']["edge"]
    edges = list(zip(stream_tree_edge[:, 0], stream_tree_edge[:, 1]))
    stream_tree.add_edges_from(edges)

    dict_edge_shift_dist = dict()
    # maximum distance from cells to branch
    max_dist = np.percentile(adata.obs['epg_edge_dist'], dist_pctl)

    leaves = [k for k, v in stream_tree.degree() if v == 1]
    n_nonroot_leaves = len(list(set(leaves) - set([source])))
    dict_bfs_pre = dict(nx.bfs_predecessors(stream_tree, source))
    dict_bfs_suc = dict(nx.bfs_successors(stream_tree, source))
    # depth first search
    dfs_nodes = _dfs_nodes_modified(stream_tree, source, preference=preference)
    dfs_nodes_copy = deepcopy(dfs_nodes)
    id_leaf = 0
    while len(dfs_nodes_copy) > 1:
        node = dfs_nodes_copy.pop()
        pre_node = dict_bfs_pre[node]
        if node in leaves:
            dict_edge_shift_dist[(pre_node, node)] = \
                2*max_dist*(id_leaf-(n_nonroot_leaves/2.0))
            id_leaf = id_leaf+1
        else:
            suc_nodes = dict_bfs_suc[node]
            dict_edge_shift_dist[(pre_node, node)] = \
                (sum([dict_edge_shift_dist[(node, sn)]
                 for sn in suc_nodes]))/float(len(suc_nodes))
    return dict_edge_shift_dist


def _get_streamtree_id(adata):
    """convert epg edge id to stream tree edge id."""
    dict_epg_to_st = dict()
    for i, x in enumerate(adata.uns['epg']['edge']):
        for j, y in enumerate(adata.uns['stream_tree']['edge_nodes']):
            if set(x) <= set(y):
                dict_epg_to_st[i] = j
                break
    df_st = adata.obs['epg_edge_id'].map(dict_epg_to_st)
    return df_st


def _add_stream_sc_pos(
        adata,
        source=0,
        dist_scale=1,
        dist_pctl=95,
        preference=None,
        key='epg'
):

    _construct_stream_tree(adata, source=source, key=key)
    if source not in adata.uns['stream_tree']['node']:
        raise ValueError(f"There is no source {source}")
    stream_tree = nx.Graph()
    stream_tree_edge = adata.uns['stream_tree']["edge"]
    stream_tree_edge_len = adata.uns['stream_tree']["edge_len"]
    edges_weighted = list(zip(
        stream_tree_edge[:, 0],
        stream_tree_edge[:, 1],
        stream_tree_edge_len))
    stream_tree.add_weighted_edges_from(edges_weighted, weight='len')

    dict_bfs_pre = dict(nx.bfs_predecessors(stream_tree, source))
    dict_bfs_suc = dict(nx.bfs_successors(stream_tree, source))
    dict_edge_shift_dist = \
        _calculate_shift_distance(
            adata, source=source,
            dist_pctl=dist_pctl, preference=preference)
    dict_path_len = nx.shortest_path_length(
        stream_tree, source=source, weight='len')
    df_st_id = _get_streamtree_id(adata)
    pseudotime = infer_pseudotime(adata, source=source, copy=True)
    df_cells_pos = pd.DataFrame(index=adata.obs.index, columns=['cell_pos'])
    dict_edge_pos = {}
    dict_node_pos = {}
    adata.uns['stream_tree']['edge_pos'] = dict()
    for i, edge in enumerate(stream_tree_edge):
        node_pos_st = np.array(
            [dict_path_len[edge[0]], dict_edge_shift_dist[tuple(edge)]])
        node_pos_ed = np.array(
            [dict_path_len[edge[1]], dict_edge_shift_dist[tuple(edge)]])
        id_cells = np.where(df_st_id == i)[0]
        cells_pos_x = pseudotime[id_cells]
        np.random.seed(100)
        cells_pos_y = node_pos_st[1] \
            + dist_scale*adata.obs[f'{key}_edge_dist'].iloc[id_cells]\
            * np.random.choice([1, -1], size=id_cells.shape[0])
        cells_pos = np.array((cells_pos_x, cells_pos_y)).T
        df_cells_pos.iloc[id_cells, 0] = \
            [cells_pos[i, :].tolist() for i in range(cells_pos.shape[0])]
        dict_edge_pos[tuple(edge)] = np.array([node_pos_st, node_pos_ed])
        if(edge[0] not in dict_bfs_pre.keys()):
            dict_node_pos[edge[0]] = node_pos_st
        dict_node_pos[edge[1]] = node_pos_ed
        edge_pos = dict_edge_pos[tuple(edge)]
        if edge[0] in dict_bfs_pre.keys():
            pre_node = dict_bfs_pre[edge[0]]
            link_edge_pos = np.array(
                [dict_edge_pos[(pre_node, edge[0])][1, ],
                 dict_edge_pos[tuple(edge)][0, ]])
            edge_pos = np.vstack((link_edge_pos, edge_pos))
        adata.uns['stream_tree']['edge_pos'][tuple(edge)] = edge_pos

    adata.uns['stream_tree']['cell_pos'] = \
        np.array(df_cells_pos['cell_pos'].tolist())

    if stream_tree.degree(source) > 1:
        suc_nodes = dict_bfs_suc[source]
        edges = [(source, sn) for sn in suc_nodes]
        edges_y_pos = [dict_edge_pos[tuple(x)][0, 1] for x in edges]
        max_y_pos = max(edges_y_pos)
        min_y_pos = min(edges_y_pos)
        median_y_pos = np.median(edges_y_pos)
        x_pos = dict_edge_pos[edges[0]][0, 0]
        dict_node_pos[source] = np.array([x_pos, median_y_pos])
        link_edge_pos = np.array([[x_pos, min_y_pos], [x_pos, max_y_pos]])
        adata.uns['stream_tree']['edge_pos'][(source, source)] = link_edge_pos

    adata.uns['stream_tree']['node_pos'] = \
        np.array([dict_node_pos[x] for x in adata.uns['stream_tree']['node']])


def _arrowed_spines(
        ax,
        x_width_fraction=0.03,
        x_height_fraction=0.02,
        lw=None,
        ohg=0.2,
        locations=('bottom right', 'left up'),
        **arrow_kwargs
):
    """
    Add arrows to the requested spines
    Code originally sourced here:
    https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/ # noqa
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448 # noqa
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: 
        The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. 
        If not supplied, default behaviour is to use the value on the current bottom spine.
        ('width' in ax.arrow() is actually controling the line width)
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". 
        These must be orthogonal (e.g. "left left" will result in an error).
        Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        lw = ax.spines['bottom'].get_linewidth()*1e-4

    annots = {}
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)*height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)*width/height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, \
            "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, \
            "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError(
                    "Only left/right arrows supported on the bottom and top")
            dy = 0
            head_width = hw
            head_length = hl
            y = ymin if side == 'bottom' else ymax
            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax
        else:
            if direction in {'left', 'right'}:
                raise ValueError(
                    "Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl
            x = xmin if side == 'left' else xmax
            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax
        annots[loc_str] = ax.arrow(
            x, y, dx, dy,
            fc='k', ec='k', width=lw,
            head_width=head_width,
            head_length=head_length,
            **arrow_kwargs)
    return annots
