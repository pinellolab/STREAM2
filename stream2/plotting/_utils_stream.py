"""Utility functions for stream plots."""

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_string_dtype,
    is_numeric_dtype
)
import networkx as nx
from copy import deepcopy
import itertools
from scipy import interpolate
from scipy.signal import savgol_filter
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


# modified depth first search for nodes
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


# modified depth first search for edges
def _dfs_edges_modified(tree, source, preference=None):
    visited, queue = [], [source]
    bfs_tree = nx.bfs_tree(tree, source=source)
    predecessors = dict(nx.bfs_predecessors(bfs_tree, source))
    edges = []
    while queue:
        vertex = queue.pop()
        if vertex not in visited:
            visited.append(vertex)
            if vertex in predecessors.keys():
                edges.append((predecessors[vertex], vertex))
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
            queue.extend(unvisited)
    return edges


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


def _get_streamtree_edge_id(adata):
    """convert epg edge id to stream tree edge id."""
    dict_epg_to_st = dict()
    for i, x in enumerate(adata.uns['epg']['edge']):
        for j, y in enumerate(adata.uns['stream_tree']['edge_nodes']):
            if set(x) <= set(y):
                dict_epg_to_st[i] = j
                break
    df_edge_id = adata.obs['epg_edge_id'].map(dict_epg_to_st)
    df_edge_id.name = 'st_edge_id'
    return df_edge_id


def _get_streamtree_edge_loc(adata, stream_tree):
    source = adata.uns['stream_tree']['params']['source']
    key = adata.uns['stream_tree']['params']['key']
    stream_tree_edge = adata.uns['stream_tree']["edge"]
    df = pd.Series(data=infer_pseudotime(
                            adata,
                            source=source,
                            key=key,
                            copy=True),
                   index=adata.obs_names,
                   name='st_edge_loc')
    for edge in stream_tree_edge:
        id_cells = stream_tree.edges[edge]['cells']
        nodes_pre = nx.shortest_path(
            stream_tree, source=source, target=edge[0])
        len_pre = 0
        for edge_pre in list(zip(nodes_pre[:-1], nodes_pre[1:])):
            len_pre += stream_tree.edges[edge_pre]['len']
        df.iloc[id_cells] = df.iloc[id_cells] - len_pre
    return df


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
    df_st_id = _get_streamtree_edge_id(adata)
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
        if edge[0] not in dict_bfs_pre.keys():
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


def _find_root_to_leaf_paths(stream_tree, source):
    list_paths = list()
    for x in stream_tree.nodes():
        if (x != source) & (stream_tree.degree(x) == 1):
            path = list(nx.all_simple_paths(stream_tree, source, x))[0]
            list_edges = list()
            for ft, sd in zip(path, path[1:]):
                list_edges.append((ft, sd))
            list_paths.append(list_edges)
    return list_paths


def _find_longest_path(list_paths, len_ori):
    list_lengths = list()
    for x in list_paths:
        list_lengths.append(sum([len_ori[x_i] for x_i in x]))
    return max(list_lengths)


# find all paths
def _find_paths(dict_tree, dfs_nodes):
    dict_paths_top = dict()
    dict_paths_base = dict()
    for node_i in dfs_nodes:
        prev_node = dict_tree[node_i]['prev']
        next_nodes = dict_tree[node_i]['next']
        if (prev_node == '') or (len(next_nodes) > 1):
            if prev_node == '':
                cur_node_top = node_i
                cur_node_base = node_i
                stack_top = [cur_node_top]
                stack_base = [cur_node_base]
                while len(dict_tree[cur_node_top]['next']) > 0:
                    cur_node_top = dict_tree[cur_node_top]['next'][0]
                    stack_top.append(cur_node_top)
                dict_paths_top[(node_i, next_nodes[0])] = stack_top
                while len(dict_tree[cur_node_base]['next']) > 0:
                    cur_node_base = dict_tree[cur_node_base]['next'][-1]
                    stack_base.append(cur_node_base)
                dict_paths_base[(node_i, next_nodes[-1])] = stack_base
            for i_mid in range(len(next_nodes)-1):
                cur_node_base = next_nodes[i_mid]
                cur_node_top = next_nodes[i_mid+1]
                stack_base = [node_i, cur_node_base]
                stack_top = [node_i, cur_node_top]
                while len(dict_tree[cur_node_base]['next']) > 0:
                    cur_node_base = dict_tree[cur_node_base]['next'][-1]
                    stack_base.append(cur_node_base)
                dict_paths_base[(node_i, next_nodes[i_mid])] = stack_base
                while len(dict_tree[cur_node_top]['next']) > 0:
                    cur_node_top = dict_tree[cur_node_top]['next'][0]
                    stack_top.append(cur_node_top)
                dict_paths_top[(node_i, next_nodes[i_mid+1])] = stack_top
    return dict_paths_top, dict_paths_base


def _cal_stream_polygon_string(
    adata,
    dict_ann,
    source=0,
    preference=None,
    dist_scale=0.9,
    factor_num_win=10,
    factor_min_win=2.0,
    factor_width=2.5,
    log_scale=False,
    factor_zoomin=100.0,
    key='epg'
):
    list_ann_string = [k for k, v in dict_ann.items() if is_string_dtype(v)]
    if 'stream_tree' not in adata.uns_keys():
        if adata.uns['stream_tree']['params']['source'] != source:
            _construct_stream_tree(adata, source=source, key=key)
    else:
        _construct_stream_tree(adata, source=source, key=key)

    stream_tree = nx.Graph()
    stream_tree_edge = adata.uns['stream_tree']["edge"]
    stream_tree_edge_len = adata.uns['stream_tree']["edge_len"]
    edges_weighted = list(zip(
        stream_tree_edge[:, 0],
        stream_tree_edge[:, 1],
        stream_tree_edge_len))
    stream_tree.add_weighted_edges_from(edges_weighted, weight="len")

    df_st_edge_id = _get_streamtree_edge_id(adata)
    dict_edge_cells = {}
    for i, edge in enumerate(stream_tree_edge):
        id_cells = np.where(df_st_edge_id == i)[0]
        dict_edge_cells[tuple(edge)] = {"cells": id_cells}
    nx.set_edge_attributes(stream_tree, dict_edge_cells)

    df_st_edge_loc = _get_streamtree_edge_loc(adata, stream_tree)

    df_stream = pd.concat([df_st_edge_id, df_st_edge_loc], axis=1)
    df_stream = df_stream.astype('object')
    df_stream['st_edge'] = ''
    for i, edge in enumerate(stream_tree_edge):
        id_cells = np.where(df_st_edge_id == i)[0]
        df_stream['st_edge'].iloc[id_cells] = \
            pd.Series(
                index=df_stream.index[id_cells],
                data=[tuple(edge)] * len(id_cells))

    dict_verts = dict()  # coordinates of all vertices
    dict_extent = dict()  # the extent of plot

    dfs_edges = _dfs_edges_modified(
        stream_tree, source, preference=preference)
    dfs_nodes = list(dict.fromkeys(sum(dfs_edges, ())))

    len_ori = {stream_tree.edges[x]['len'] for x in stream_tree_edge}
    bfs_prev = dict(nx.bfs_predecessors(stream_tree, source))
    bfs_next = dict(nx.bfs_successors(stream_tree, source))
    dict_tree = {}
    for x in dfs_nodes:
        dict_tree[x] = {'prev': "", 'next': []}
        if x in bfs_prev.keys():
            dict_tree[x]['prev'] = bfs_prev[x]
        if x in bfs_next.keys():
            x_rank = [dfs_nodes.index(x_next) for x_next in bfs_next[x]]
            dict_tree[x]['next'] = [
                y for _, y in sorted(
                    zip(x_rank, bfs_next[x]), key=lambda y: y[0])]

    dict_shift_dist = dict()  # shifting distance of each branch
    leaves = [n for n, d in stream_tree.degree() if d == 1]
    id_leaf = 0
    dfs_nodes_copy = deepcopy(dfs_nodes)
    num_nonroot_leaf = len(list(set(leaves) - set([source])))
    while len(dfs_nodes_copy) > 1:
        node = dfs_nodes_copy.pop()
        prev_node = dict_tree[node]['prev']
        if node in leaves:
            dict_shift_dist[(prev_node, node)] = \
                -(float(1)/dist_scale)*(num_nonroot_leaf-1)/2.0 \
                + id_leaf*(float(1)/dist_scale)
            id_leaf = id_leaf+1
        else:
            next_nodes = dict_tree[node]['next']
            dict_shift_dist[(prev_node, node)] = \
                (sum([dict_shift_dist[(node, next_node)]
                      for next_node in next_nodes]))/float(len(next_nodes))
    if stream_tree.degree(source) > 1:
        next_nodes = dict_tree[source]['next']
        dict_shift_dist[(source, source)] = \
            (sum([dict_shift_dist[(source, next_node)]
                  for next_node in next_nodes]))/float(len(next_nodes))

    for ann in list_ann_string:
        df_stream[ann] = dict_ann[ann]
        # dataframe of bins
        df_bins = pd.DataFrame(
            index=list(df_stream[ann].unique())
            + ['boundary', 'center', 'edge'])
        list_paths = _find_root_to_leaf_paths(stream_tree, source)
        max_path_len = _find_longest_path(list_paths, len_ori)
        size_w = max_path_len/float(factor_num_win)
        if size_w > min(len_ori.values())/float(factor_min_win):
            size_w = min(len_ori.values())/float(factor_min_win)
        # step of sliding window (the divisor should be an even number)
        step_w = size_w/2
        if len(dict_shift_dist) > 1:
            max_width = (max_path_len/float(factor_width))\
                / (max(dict_shift_dist.values())
                   - min(dict_shift_dist.values()))
        else:
            max_width = max_path_len/float(factor_width)
        dict_shift_dist = {
            x: dict_shift_dist[x]*max_width for x in dict_shift_dist.keys()}
        min_width = 0.0  # min width of branch
        min_cellnum = 0  # the minimal cell number in one branch
        min_bin_cellnum = 0  # the minimal cell number in each bin
        # filter out cells whose total count
        # on one edge is below the min_cellnum
        dict_edge_filter = dict()
        df_edge_cellnum = pd.DataFrame(
            index=df_stream[ann].unique(),
            columns=dfs_edges,
            dtype=float)
        for i, edge_i in enumerate(dfs_edges):
            df_edge_i = df_stream[df_stream['st_edge'] == edge_i]
            cells_kept = df_edge_i[ann].value_counts()[
                df_edge_i[ann].value_counts() > min_cellnum].index
            df_edge_i = df_edge_i[df_edge_i[ann].isin(cells_kept)]
            dict_edge_filter[edge_i] = df_edge_i
            for cell_i in df_stream[ann].unique():
                df_edge_cellnum[edge_i][cell_i] = float(
                    df_edge_i[df_edge_i[ann] == cell_i].shape[0])
        for i, edge_i in enumerate(dfs_edges):
            # degree of the start node
            degree_st = stream_tree.degree(edge_i[0])
            # degree of the end node
            degree_end = stream_tree.degree(edge_i[1])
            # matrix of windows only appearing on one edge
            mat_w = np.vstack([
                np.arange(0, len_ori[edge_i] - size_w
                          + (len_ori[edge_i]/10**6), step_w),
                np.arange(size_w, len_ori[edge_i]
                          + (len_ori[edge_i]/10**6), step_w)]).T
            mat_w[-1, -1] = len_ori[edge_i]
            if degree_st == 1:
                mat_w = np.insert(mat_w, 0, [0, size_w/2.0], axis=0)
            if degree_end == 1:
                mat_w = np.insert(
                    mat_w, mat_w.shape[0],
                    [len_ori[edge_i]-size_w/2.0, len_ori[edge_i]],
                    axis=0)
            total_bins = df_bins.shape[1]  # current total number of bins

            if degree_st > 1 and i == 0:
                # avoid warning "DataFrame is highly fragmented."
                df_bins = df_bins.copy()
                # matrix of windows spanning multiple edges
                mat_w_common = np.array([[0, size_w/2.0], [0, size_w]])
                # neighbor nodes
                nb_nodes = list(stream_tree.neighbors(edge_i[0]))
                index_nb_nodes = [dfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = \
                    np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                # matrix of windows spanning multiple edges
                total_bins = df_bins.shape[1]
                for i_win in range(mat_w_common.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],
                                "win"+str(total_bins+i_win)] = 0
                    df_bins.at['edge', "win"+str(total_bins+i_win)] = \
                        [(source, source)]
                    for j in range(degree_st):
                        df_edge_j = dict_edge_filter[(edge_i[0], nb_nodes[j])]
                        cell_num_common2 = \
                            df_edge_j[np.logical_and(
                                df_edge_j['st_edge_loc'] >= 0,
                                df_edge_j['st_edge_loc'] <= mat_w_common[
                                    i_win, 1])][ann].value_counts()
                        df_bins.loc[cell_num_common2.index,
                                    "win"+str(total_bins + i_win)] = \
                            df_bins.loc[cell_num_common2.index,
                                        "win"+str(total_bins + i_win)]\
                            + cell_num_common2
                        df_bins.loc[
                            'edge',
                            "win"+str(total_bins+i_win)].append(
                                (edge_i[0], nb_nodes[j]))
                    df_bins.at['boundary', "win"+str(total_bins+i_win)] = \
                        mat_w_common[i_win, :]
                    if i_win == 0:
                        df_bins.loc['center', "win"+str(total_bins+i_win)] = 0
                    else:
                        df_bins.loc['center',
                                    "win"+str(total_bins+i_win)] = size_w/2

            # the maximal number of merging bins
            max_binnum = np.around((len_ori[edge_i]/4.0-size_w)/step_w)
            df_edge_i = dict_edge_filter[edge_i]
            total_bins = df_bins.shape[1]  # current total number of bins

            if max_binnum <= 1:
                df_bins = df_bins.copy()
                for i_win in range(mat_w.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],
                                "win"+str(total_bins+i_win)] = 0
                    cell_num = df_edge_i[np.logical_and(
                        df_edge_i['st_edge_loc']
                        >= mat_w[i_win, 0],
                        df_edge_i['st_edge_loc']
                        <= mat_w[i_win, 1])][ann].value_counts()
                    df_bins.loc[
                        cell_num.index,
                        "win"+str(total_bins+i_win)] = cell_num
                    df_bins.at[
                        'boundary',
                        "win"+str(total_bins+i_win)] = mat_w[i_win, :]
                    if degree_st == 1 and i_win == 0:
                        df_bins.loc['center', "win"+str(total_bins+i_win)] = 0
                    elif degree_end == 1 and i_win == (mat_w.shape[0]-1):
                        df_bins.loc[
                            'center',
                            "win"+str(total_bins+i_win)] = len_ori[edge_i]
                    else:
                        df_bins.loc[
                            'center',
                            "win"+str(total_bins+i_win)] =\
                                np.mean(mat_w[i_win, :])
                col_wins = [
                    "win"+str(total_bins+i_win)
                    for i_win in range(mat_w.shape[0])]
                df_bins.loc['edge', col_wins] = pd.Series(
                    index=col_wins,
                    data=[[edge_i]]*len(col_wins))

            if max_binnum > 1:
                df_bins = df_bins.copy()
                id_stack = []
                for i_win in range(mat_w.shape[0]):
                    id_stack.append(i_win)
                    bd_bins = [
                        mat_w[id_stack[0], 0],
                        mat_w[id_stack[-1], 1]]  # boundary of merged bins
                    cell_num = df_edge_i[np.logical_and(
                        df_edge_i['st_edge_loc'] >=
                        bd_bins[0],
                        df_edge_i['st_edge_loc'] <=
                        bd_bins[1])][ann].value_counts()
                    if len(id_stack) == max_binnum \
                            or any(cell_num > min_bin_cellnum) \
                            or i_win == mat_w.shape[0]-1:
                        df_bins["win"+str(total_bins)] = ""
                        df_bins.loc[
                            df_bins.index[:-3],
                            "win"+str(total_bins)] = 0
                        df_bins.loc[
                            cell_num.index,
                            "win"+str(total_bins)] = cell_num
                        df_bins.at['boundary', "win"+str(total_bins)] = bd_bins
                        df_bins.at['edge', "win"+str(total_bins)] = [edge_i]
                        if degree_st == 1 and (0 in id_stack):
                            df_bins.loc['center', "win"+str(total_bins)] = 0
                        elif degree_end == 1 and i_win == (mat_w.shape[0]-1):
                            df_bins.loc[
                                'center',
                                "win"+str(total_bins)] = len_ori[edge_i]
                        else:
                            df_bins.loc[
                                'center',
                                "win"+str(total_bins)] = np.mean(bd_bins)
                        total_bins = total_bins + 1
                        id_stack = []

            if degree_end > 1:
                df_bins = df_bins.copy()
                # matrix of windows appearing on multiple edges
                mat_w_common = np.vstack(
                    [np.arange(
                        len_ori[edge_i] - size_w + step_w,
                        len_ori[edge_i] + (len_ori[edge_i]/10**6),
                        step_w),
                     np.arange(
                        step_w,
                        size_w+(len_ori[edge_i]/10**6),
                        step_w)]).T
                # neighbor nodes
                nb_nodes = list(stream_tree.neighbors(edge_i[1]))
                nb_nodes.remove(edge_i[0])
                index_nb_nodes = [dfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[
                    np.argsort(index_nb_nodes)].tolist()

                # matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1]  # current total number of bins
                if mat_w_common.shape[0] > 0:
                    for i_win in range(mat_w_common.shape[0]):
                        df_bins["win"+str(total_bins+i_win)] = ""
                        df_bins.loc[
                            df_bins.index[:-3],
                            "win"+str(total_bins+i_win)] = 0
                        cell_num_common1 =\
                            df_edge_i[np.logical_and(
                                df_edge_i['st_edge_loc']
                                > mat_w_common[i_win, 0],
                                df_edge_i['st_edge_loc']
                                <= len_ori[edge_i])][ann].value_counts()
                        df_bins.loc[
                            cell_num_common1.index,
                            "win"+str(total_bins+i_win)] = cell_num_common1
                        df_bins.at[
                            'edge', "win"+str(total_bins+i_win)] = [edge_i]
                        for j in range(degree_end - 1):
                            df_edge_j = dict_edge_filter[(
                                edge_i[1], nb_nodes[j])]
                            cell_num_common2 = \
                                df_edge_j[np.logical_and(
                                    df_edge_j['st_edge_loc'] >= 0,
                                    df_edge_j['st_edge_loc']
                                    <= mat_w_common[i_win, 1]
                                    )][ann].value_counts()
                            df_bins.loc[cell_num_common2.index,
                                        "win"+str(total_bins+i_win)] = \
                                df_bins.loc[
                                    cell_num_common2.index,
                                    "win"+str(total_bins+i_win)] \
                                + cell_num_common2
                            if abs(((sum(
                                mat_w_common[i_win, :])
                                + len_ori[edge_i])/2)
                                - (len_ori[edge_i]
                                    + size_w/2.0)) < step_w/100.0:
                                df_bins.loc[
                                    'edge',
                                    "win"+str(total_bins+i_win)].append(
                                        (edge_i[1], nb_nodes[j]))
                        df_bins.at['boundary', "win"+str(total_bins+i_win)] = \
                            mat_w_common[i_win, :]
                        df_bins.loc['center', "win"+str(total_bins+i_win)] = \
                            (sum(mat_w_common[i_win, :])+len_ori[edge_i])/2

        df_bins = df_bins.copy()
        # order cell names by the index of first non-zero
        cell_list = df_bins.index[:-3]
        id_nonzero = []
        for cellname in cell_list:
            if np.flatnonzero(
                    df_bins.loc[cellname, ]).size == 0:
                print('Cell '+cellname+' does not exist')
                break
            else:
                id_nonzero.append(
                    np.flatnonzero(df_bins.loc[cellname, ])[0])
        cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()
        # original count
        df_bins_ori = df_bins.reindex(
            cell_list_sorted+['boundary', 'center', 'edge'])
        if log_scale:
            df_n_cells = df_bins_ori.iloc[:-3, :].sum()
            df_n_cells = df_n_cells/df_n_cells.max()*factor_zoomin
            df_bins_ori.iloc[:-3, :] = \
                df_bins_ori.iloc[:-3, :]*np.log2(df_n_cells+1)/(df_n_cells+1)
        df_bins_cumsum = df_bins_ori.copy()
        df_bins_cumsum.iloc[:-3, :] = \
            df_bins_ori.iloc[:-3, :][::-1].cumsum()[::-1]

        # normalization
        df_bins_cumsum_norm = df_bins_cumsum.copy()
        df_bins_cumsum_norm.iloc[:-3, :] = \
            min_width + max_width*(
                df_bins_cumsum.iloc[:-3, :])\
            / (df_bins_cumsum.iloc[:-3, :]).values.max()

        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3, :] = \
            df_bins_cumsum_norm.iloc[:-3, :].subtract(
                df_bins_cumsum_norm.iloc[0, :]/2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4, :] = df_bins_top.iloc[1:-3, ].values
        df_bins_base.iloc[-4, :] = 0-df_bins_cumsum_norm.iloc[0, :]/2.0
        dict_forest = {
            cellname: {nodename: {'prev': "", 'next': "", 'div': ""}
                       for nodename in dfs_nodes}
            for cellname in df_edge_cellnum.index}
        for cellname in cell_list_sorted:
            for node_i in dfs_nodes:
                nb_nodes = list(stream_tree.neighbors(node_i))
                index_in_bfs = [dfs_nodes.index(nb) for nb in nb_nodes]
                nb_nodes_sorted = np.array(
                    nb_nodes)[np.argsort(index_in_bfs)].tolist()
                if node_i == source:
                    next_nodes = nb_nodes_sorted
                    prev_nodes = ''
                else:
                    next_nodes = nb_nodes_sorted[1:]
                    prev_nodes = nb_nodes_sorted[0]
                dict_forest[cellname][node_i]['next'] = next_nodes
                dict_forest[cellname][node_i]['prev'] = prev_nodes
                if len(next_nodes) > 1:
                    pro_next_edges = []  # proportion of next edges
                    for nt in next_nodes:
                        id_wins = [ix for ix, x in enumerate(
                            df_bins_cumsum_norm.loc['edge', :])
                            if x == [(node_i, nt)]]
                        pro_next_edges.append(
                            df_bins_cumsum_norm.loc[
                                cellname, 'win'+str(id_wins[0])])
                    if sum(pro_next_edges) == 0:
                        dict_forest[cellname][node_i]['div'] = \
                            np.cumsum(np.repeat(1.0/len(next_nodes),
                                      len(next_nodes))).tolist()
                    else:
                        dict_forest[cellname][node_i]['div'] = \
                            (np.cumsum(
                                pro_next_edges)/sum(
                                    pro_next_edges)).tolist()

        # Shift
        # coordinates of end points
        dict_ep_top = {cellname: dict() for cellname in cell_list_sorted}
        dict_ep_base = {cellname: dict() for cellname in cell_list_sorted}
        # center coordinates of end points in each branch
        dict_ep_center = dict()

        df_top_x = df_bins_top.copy()  # x coordinates in top line
        df_top_y = df_bins_top.copy()  # y coordinates in top line
        df_base_x = df_bins_base.copy()  # x coordinates in base line
        df_base_y = df_bins_base.copy()  # y coordinates in base line

        for edge_i in dfs_edges:
            id_wins = [i for i, x in enumerate(
                df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
            prev_node = dict_tree[edge_i[0]]['prev']
            if prev_node == '':
                x_st = 0
                if stream_tree.degree(source) > 1:
                    id_wins = id_wins[1:]
            else:
                id_wins = id_wins[1:]  # remove the overlapped window
                x_st = dict_ep_center[(prev_node, edge_i[0])][0] - step_w
            y_st = dict_shift_dist[edge_i]
            for cellname in cell_list_sorted:
                # top line
                px_top = df_bins_top.loc[
                    'center', list(map(lambda x: 'win' + str(x), id_wins))]
                py_top = df_bins_top.loc[
                    cellname, list(map(lambda x: 'win' + str(x), id_wins))]
                px_top_prime = x_st + px_top
                py_top_prime = y_st + py_top
                dict_ep_top[cellname][edge_i] = \
                    [px_top_prime[-1], py_top_prime[-1]]
                df_top_x.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins))]\
                    = px_top_prime
                df_top_y.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins))]\
                    = py_top_prime
                # base line
                px_base = df_bins_base.loc[
                    'center',
                    list(map(lambda x: 'win' + str(x), id_wins))]
                py_base = df_bins_base.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins))]
                px_base_prime = x_st + px_base
                py_base_prime = y_st + py_base
                dict_ep_base[cellname][edge_i] = \
                    [px_base_prime[-1], py_base_prime[-1]]
                df_base_x.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins))]\
                    = px_base_prime
                df_base_y.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins))]\
                    = py_base_prime
            dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

        id_wins_start = [i for i, x in enumerate(
            df_bins_cumsum_norm.loc['edge', :])
            if x[0] == (source, source)]
        if len(id_wins_start) > 0:
            mean_shift_dist =\
                np.mean([dict_shift_dist[(source, x)]
                        for x in dict_forest[
                            cell_list_sorted[0]][source]['next']])
            for cellname in cell_list_sorted:
                # top line
                px_top = df_bins_top.loc[
                    'center',
                    list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_top = df_bins_top.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_top_prime = 0 + px_top
                py_top_prime = mean_shift_dist + py_top
                df_top_x.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins_start))]\
                    = px_top_prime
                df_top_y.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins_start))]\
                    = py_top_prime
                # base line
                px_base = df_bins_base.loc[
                    'center',
                    list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_base = df_bins_base.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_base_prime = 0 + px_base
                py_base_prime = mean_shift_dist + py_base
                df_base_x.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x),
                             id_wins_start))] = px_base_prime
                df_base_y.loc[
                    cellname,
                    list(map(lambda x: 'win' + str(x),
                             id_wins_start))] = py_base_prime

        # determine joints points
        # coordinates of joint points
        dict_joint_top = {
            cellname: dict() for cellname in cell_list_sorted}
        dict_joint_base = {
            cellname: dict() for cellname in cell_list_sorted}
        if stream_tree.degree(source) == 1:
            id_joints = [i for i, x in enumerate(
                df_bins_cumsum_norm.loc['edge', :]) if len(x) > 1]
        else:
            id_joints = [i for i, x in enumerate(
                df_bins_cumsum_norm.loc['edge', :])
                if len(x) > 1 and x[0] != (source, source)]
            id_joints.insert(0, 1)
        for id_j in id_joints:
            joint_edges = df_bins_cumsum_norm.loc['edge', 'win'+str(id_j)]
            for id_div, edge_i in enumerate(joint_edges[1:]):
                id_wins = [i for i, x in enumerate(
                    df_bins_cumsum_norm.loc['edge', :]) if x == [edge_i]]
                for cellname in cell_list_sorted:
                    if len(dict_forest[cellname][edge_i[0]]['div']) > 0:
                        prev_node_top_x = df_top_x.loc[
                            cellname, 'win'+str(id_j)]
                        prev_node_top_y = df_top_y.loc[
                            cellname, 'win'+str(id_j)]
                        prev_node_base_x = df_base_x.loc[
                            cellname, 'win'+str(id_j)]
                        prev_node_base_y = df_base_y.loc[
                            cellname, 'win'+str(id_j)]
                        div = dict_forest[cellname][edge_i[0]]['div']
                        if id_div == 0:
                            px_top_prime_st = prev_node_top_x
                            py_top_prime_st = prev_node_top_y
                        else:
                            px_top_prime_st = prev_node_top_x \
                                + (prev_node_base_x - prev_node_top_x)\
                                * div[id_div-1]
                            py_top_prime_st = prev_node_top_y \
                                + (prev_node_base_y - prev_node_top_y)\
                                * div[id_div-1]
                        px_base_prime_st = prev_node_top_x \
                            + (prev_node_base_x - prev_node_top_x)*div[id_div]
                        py_base_prime_st = prev_node_top_y \
                            + (prev_node_base_y - prev_node_top_y)*div[id_div]
                        df_top_x.loc[cellname,
                                     'win'+str(id_wins[0])] = px_top_prime_st
                        df_top_y.loc[cellname,
                                     'win'+str(id_wins[0])] = py_top_prime_st
                        df_base_x.loc[cellname,
                                      'win'+str(id_wins[0])] = px_base_prime_st
                        df_base_y.loc[cellname,
                                      'win'+str(id_wins[0])] = py_base_prime_st
                        dict_joint_top[cellname][edge_i] =\
                            np.array([px_top_prime_st, py_top_prime_st])
                        dict_joint_base[cellname][edge_i] =\
                            np.array([px_base_prime_st, py_base_prime_st])

        dict_tree_copy = deepcopy(dict_tree)
        dict_paths_top, dict_paths_base = \
            _find_paths(dict_tree_copy, dfs_nodes)

        # identify boundary of each edge
        dict_edge_bd = dict()
        for edge_i in dfs_edges:
            id_wins = [i for i, x in enumerate(
                df_top_x.loc['edge', :]) if edge_i in x]
            dict_edge_bd[edge_i] = [
                df_top_x.iloc[0, id_wins[0]], df_top_x.iloc[0, id_wins[-1]]]

        x_smooth = np.unique(np.arange(
            min(df_base_x.iloc[0, :]),
            max(df_base_x.iloc[0, :]),
            step=step_w/20).tolist() + [max(df_base_x.iloc[0, :])]).tolist()
        x_joints = df_top_x.iloc[0, id_joints].tolist()
        # replace nearest value in x_smooth by x axis of joint points
        for x in x_joints:
            x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

        dict_smooth_linear = {cellname: {
            'top': dict(), 'base': dict()} for cellname in cell_list_sorted}
        # interpolation
        for edge_i_top in dict_paths_top.keys():
            path_i_top = dict_paths_top[edge_i_top]
            id_wins_top = [i_x for i_x, x in enumerate(
                df_top_x.loc['edge']) if set(
                    np.unique(x)).issubset(set(path_i_top))]
            if stream_tree.degree(source) > 1 \
                and edge_i_top == (
                    source, dict_forest[
                        cell_list_sorted[0]][source]['next'][0]):
                id_wins_top.insert(0, 1)
                id_wins_top.insert(0, 0)
            for cellname in cell_list_sorted:
                x_top = df_top_x.loc[cellname, list(map(
                    lambda x: 'win' + str(x), id_wins_top))].tolist()
                y_top = df_top_y.loc[cellname, list(map(
                    lambda x: 'win' + str(x), id_wins_top))].tolist()
                f_top_linear = interpolate.interp1d(
                    x_top, y_top, kind='linear')
                x_top_new = [x for x in x_smooth
                             if (x >= x_top[0]) and (x <= x_top[-1])] \
                    + [x_top[-1]]
                x_top_new = np.unique(x_top_new).tolist()
                y_top_new_linear = f_top_linear(x_top_new)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node], path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(x_top_new)
                                   if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_linear[cellname]['top'][edge_i] = \
                        pd.DataFrame([
                            np.array(x_top_new)[id_selected],
                            np.array(y_top_new_linear)[id_selected]],
                                     index=['x', 'y'])
        for edge_i_base in dict_paths_base.keys():
            path_i_base = dict_paths_base[edge_i_base]
            id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge'])
                            if set(np.unique(x)).issubset(set(path_i_base))]
            if stream_tree.degree(source) > 1 \
                and edge_i_base == (source, dict_forest[
                    cell_list_sorted[0]][source]['next'][-1]):
                id_wins_base.insert(0, 1)
                id_wins_base.insert(0, 0)
            for cellname in cell_list_sorted:
                x_base = df_base_x.loc[cellname, list(map(
                    lambda x: 'win' + str(x), id_wins_base))].tolist()
                y_base = df_base_y.loc[cellname, list(map(
                    lambda x: 'win' + str(x), id_wins_base))].tolist()
                f_base_linear = interpolate.interp1d(
                    x_base, y_base, kind='linear')
                x_base_new = [x for x in x_smooth
                              if (x >= x_base[0]) and (x <= x_base[-1])] \
                    + [x_base[-1]]
                x_base_new = np.unique(x_base_new).tolist()
                y_base_new_linear = f_base_linear(x_base_new)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node], path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(
                        x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_linear[cellname]['base'][edge_i] = \
                        pd.DataFrame([
                            np.array(x_base_new)[id_selected],
                            np.array(y_base_new_linear)[id_selected]],
                            index=['x', 'y'])

        # searching for edges on which cell exists
        # based on the linear interpolation
        dict_edges_CE = {cellname: [] for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in dfs_edges:
                if sum(abs(
                        dict_smooth_linear[
                            cellname]['top'][edge_i].loc['y']
                        - dict_smooth_linear[
                            cellname]['base'][edge_i].loc['y'])
                        > 1e-12):
                    dict_edges_CE[cellname].append(edge_i)

        # determine paths where cell exists
        dict_paths_CE_top = {cellname: {} for cellname in cell_list_sorted}
        dict_paths_CE_base = {cellname: {} for cellname in cell_list_sorted}
        dict_forest_CE = dict()
        for cellname in cell_list_sorted:
            edges_cn = dict_edges_CE[cellname]
            nodes = [nodename for nodename in dfs_nodes
                     if nodename in set(itertools.chain(*edges_cn))]
            dict_forest_CE[cellname] = {
                nodename: {'prev': "", 'next': []} for nodename in nodes}
            for node_i in nodes:
                prev_node = dict_tree[node_i]['prev']
                if (prev_node, node_i) in edges_cn:
                    dict_forest_CE[cellname][node_i]['prev'] = prev_node
                next_nodes = dict_tree[node_i]['next']
                for x in next_nodes:
                    if (node_i, x) in edges_cn:
                        (dict_forest_CE[cellname][node_i]['next']).append(x)
            dict_paths_CE_top[cellname], dict_paths_CE_base[cellname] = \
                _find_paths(dict_forest_CE[cellname], nodes)

        dict_smooth_new = deepcopy(dict_smooth_linear)
        for cellname in cell_list_sorted:
            paths_CE_top = dict_paths_CE_top[cellname]
            for edge_i_top in paths_CE_top.keys():
                path_i_top = paths_CE_top[edge_i_top]
                edges_top = [x for x in dfs_edges if set(
                    np.unique(x)).issubset(set(path_i_top))]
                id_wins_top = [i_x for i_x, x in enumerate(
                    df_top_x.loc['edge']) if set(
                        np.unique(x)).issubset(set(path_i_top))]

                x_top = []
                y_top = []
                for e_t in edges_top:
                    if e_t == edges_top[-1]:
                        py_top_linear = dict_smooth_linear[
                            cellname]['top'][e_t].loc['y']
                        px = dict_smooth_linear[
                            cellname]['top'][e_t].loc['x']
                    else:
                        py_top_linear = dict_smooth_linear[
                            cellname]['top'][e_t].iloc[1, :-1]
                        px = dict_smooth_linear[
                            cellname]['top'][e_t].iloc[0, :-1]
                    x_top = x_top + px.tolist()
                    y_top = y_top + py_top_linear.tolist()
                x_top_new = x_top
                y_top_new = savgol_filter(y_top, 11, polyorder=1)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node], path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(
                        x_top_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_new[cellname]['top'][edge_i] = \
                        pd.DataFrame([np.array(x_top_new)[id_selected],
                                      np.array(y_top_new)[id_selected]],
                                     index=['x', 'y'])

            paths_CE_base = dict_paths_CE_base[cellname]
            for edge_i_base in paths_CE_base.keys():
                path_i_base = paths_CE_base[edge_i_base]
                edges_base = [x for x in dfs_edges if set(
                    np.unique(x)).issubset(set(path_i_base))]
                id_wins_base = [i_x for i_x, x in enumerate(
                    df_base_x.loc['edge'])
                    if set(np.unique(x)).issubset(set(path_i_base))]

                x_base = []
                y_base = []
                for e_b in edges_base:
                    if e_b == edges_base[-1]:
                        py_base_linear = dict_smooth_linear[
                            cellname]['base'][e_b].loc['y']
                        px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                    else:
                        py_base_linear = dict_smooth_linear[
                            cellname]['base'][e_b].iloc[1, :-1]
                        px = dict_smooth_linear[
                            cellname]['base'][e_b].iloc[0, :-1]
                    x_base = x_base + px.tolist()
                    y_base = y_base + py_base_linear.tolist()
                x_base_new = x_base
                y_base_new = savgol_filter(y_base, 11, polyorder=1)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node], path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(
                        x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_new[cellname]['base'][edge_i] = \
                        pd.DataFrame([
                            np.array(x_base_new)[id_selected],
                            np.array(y_base_new)[id_selected]],
                            index=['x', 'y'])
        # find all edges of polygon
        poly_edges = []
        dict_tree_copy = deepcopy(dict_tree)
        cur_node = source
        next_node = dict_tree_copy[cur_node]['next'][0]
        dict_tree_copy[cur_node]['next'].pop(0)
        poly_edges.append((cur_node, next_node))
        cur_node = next_node
        while not (next_node == source
                   and cur_node == dict_tree[source]['next'][-1]):
            while len(dict_tree_copy[cur_node]['next']) != 0:
                next_node = dict_tree_copy[cur_node]['next'][0]
                dict_tree_copy[cur_node]['next'].pop(0)
                poly_edges.append((cur_node, next_node))
                if cur_node == dict_tree[source]['next'][-1] \
                        and next_node == source:
                    break
                cur_node = next_node
            while len(dict_tree_copy[cur_node]['next']) == 0:
                next_node = dict_tree_copy[cur_node]['prev']
                poly_edges.append((cur_node, next_node))
                if cur_node == dict_tree[source]['next'][-1] \
                        and next_node == source:
                    break
                cur_node = next_node

        verts = {cellname: np.empty((0, 2))
                 for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in poly_edges:
                if edge_i in dfs_edges:
                    x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                    y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                    pxy = np.array([x_top, y_top]).T
                else:
                    edge_i = (edge_i[1], edge_i[0])
                    x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                    y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                    x_base = x_base[::-1]
                    y_base = y_base[::-1]
                    pxy = np.array([x_base, y_base]).T
                verts[cellname] = np.vstack((verts[cellname], pxy))
        dict_verts[ann] = verts

        extent = {'xmin': "", 'xmax': "", 'ymin': "", 'ymax': ""}
        for cellname in cell_list_sorted:
            for edge_i in dfs_edges:
                xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
                xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
                ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
                ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
                if extent['xmin'] == "":
                    extent['xmin'] = xmin
                else:
                    if xmin < extent['xmin']:
                        extent['xmin'] = xmin

                if extent['xmax'] == "":
                    extent['xmax'] = xmax
                else:
                    if xmax > extent['xmax']:
                        extent['xmax'] = xmax

                if extent['ymin'] == "":
                    extent['ymin'] = ymin
                else:
                    if ymin < extent['ymin']:
                        extent['ymin'] = ymin

                if extent['ymax'] == "":
                    extent['ymax'] = ymax
                else:
                    if ymax > extent['ymax']:
                        extent['ymax'] = ymax
        dict_extent[ann] = extent
    return dict_verts, dict_extent


def _fill_im_array(
    dict_im_array, df_bins_gene, stream_tree,
    df_base_x, df_base_y, df_top_y,
    xmin, xmax, ymin, ymax, im_nrow, im_ncol,
    step_w, dict_shift_dist, id_wins, edge_i,
    cellname, id_wins_prev, prev_edge
):
    pad_ratio = 0.008
    xmin_edge = df_base_x.loc[cellname, list(map(
        lambda x: 'win' + str(x), id_wins))].min()
    xmax_edge = df_base_x.loc[cellname, list(map(
        lambda x: 'win' + str(x), id_wins))].max()
    id_st_x = int(np.floor(((xmin_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
    id_ed_x = int(np.floor(((xmax_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
    if stream_tree.degree(edge_i[1]) == 1:
        id_ed_x = id_ed_x + 1
    if id_st_x < 0:
        id_st_x = 0
    if id_st_x > 0:
        id_st_x = id_st_x + 1
    if id_ed_x > (im_ncol-1):
        id_ed_x = im_ncol - 1
    if prev_edge != '':
        shift_dist = dict_shift_dist[edge_i] - dict_shift_dist[prev_edge]
        gene_color = df_bins_gene.loc[cellname, list(map(
            lambda x: 'win' + str(x), [
                id_wins_prev[-1]] + id_wins[1:]))].tolist()
    else:
        gene_color = df_bins_gene.loc[cellname, list(map(
            lambda x: 'win' + str(x), id_wins))].tolist()
    x_axis = df_base_x.loc[cellname, list(map(
        lambda x: 'win' + str(x), id_wins))].tolist()
    x_base = np.linspace(x_axis[0], x_axis[-1], id_ed_x-id_st_x+1)
    gene_color_new = np.interp(x_base, x_axis, gene_color)
    y_axis_base = df_base_y.loc[cellname, list(map(
        lambda x: 'win' + str(x), id_wins))].tolist()
    y_axis_top = df_top_y.loc[cellname, list(map(
        lambda x: 'win' + str(x), id_wins))].tolist()
    f_base_linear = interpolate.interp1d(x_axis, y_axis_base, kind='linear')
    f_top_linear = interpolate.interp1d(x_axis, y_axis_top, kind='linear')
    y_base = f_base_linear(x_base)
    y_top = f_top_linear(x_base)
    id_y_base = np.ceil((1-(y_base-ymin)/(ymax-ymin))*(im_nrow-1)).astype(int)\
        + int(im_ncol * pad_ratio)
    id_y_base[id_y_base < 0] = 0
    id_y_base[id_y_base > (im_nrow-1)] = im_nrow-1
    id_y_top = np.floor((1-(y_top-ymin)/(ymax-ymin))*(im_nrow-1)).astype(int)\
        - int(im_ncol * pad_ratio)
    id_y_top[id_y_top < 0] = 0
    id_y_top[id_y_top > (im_nrow-1)] = im_nrow-1
    id_x_base = range(id_st_x, (id_ed_x+1))
    for x in range(len(id_y_base)):
        if x in range(int(step_w/xmax * im_ncol)) and prev_edge != '':
            if shift_dist > 0:
                id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio)
                id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio)\
                    - int(abs(shift_dist)/abs(ymin - ymax) * im_nrow * 0.3)
                if id_y_top[x] < 0:
                    id_y_top[x] = 0
            if shift_dist < 0:
                id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio)\
                    + int(abs(shift_dist)/abs(ymin - ymax) * im_nrow * 0.3)
                id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio)
                if id_y_base[x] > im_nrow-1:
                    id_y_base[x] = im_nrow-1
        dict_im_array[cellname][
            id_y_top[x]:(id_y_base[x]+1), id_x_base[x]] = np.tile(
                gene_color_new[x], (id_y_base[x]-id_y_top[x]+1))
    return dict_im_array


def _cal_stream_polygon_numeric(
    adata,
    dict_ann,
    source=0,
    preference=None,
    dist_scale=0.9,
    factor_num_win=10,
    factor_min_win=2.0,
    factor_width=2.5,
    factor_nrow=200,
    factor_ncol=400,
    log_scale=False,
    factor_zoomin=100.0,
    key='epg'
):
    list_ann_numeric = [k for k, v in dict_ann.items() if is_numeric_dtype(v)]
    if 'stream_tree' not in adata.uns_keys():
        if adata.uns['stream_tree']['params']['source'] != source:
            _construct_stream_tree(adata, source=source, key=key)
    else:
        _construct_stream_tree(adata, source=source, key=key)

    stream_tree = nx.Graph()
    stream_tree_edge = adata.uns['stream_tree']["edge"]
    stream_tree_edge_len = adata.uns['stream_tree']["edge_len"]
    edges_weighted = list(zip(
        stream_tree_edge[:, 0],
        stream_tree_edge[:, 1],
        stream_tree_edge_len))
    stream_tree.add_weighted_edges_from(edges_weighted, weight="len")

    df_st_edge_id = _get_streamtree_edge_id(adata)
    dict_edge_cells = {}
    for i, edge in enumerate(stream_tree_edge):
        id_cells = np.where(df_st_edge_id == i)[0]
        dict_edge_cells[tuple(edge)] = {"cells": id_cells}
    nx.set_edge_attributes(stream_tree, dict_edge_cells)

    df_st_edge_loc = _get_streamtree_edge_loc(adata, stream_tree)

    df_stream = pd.concat([df_st_edge_id, df_st_edge_loc], axis=1)
    df_stream = df_stream.astype('object')
    df_stream['st_edge'] = ''
    for i, edge in enumerate(stream_tree_edge):
        id_cells = np.where(df_st_edge_id == i)[0]
        df_stream['st_edge'].iloc[id_cells] = \
            pd.Series(
                index=df_stream.index[id_cells],
                data=[tuple(edge)] * len(id_cells))
    df_stream['label'] = 'unknown'
    for ann in list_ann_numeric:
        df_stream[ann] = dict_ann[ann]

    dict_verts = dict()  # coordinates of all vertices

    dfs_edges = _dfs_edges_modified(
        stream_tree, source, preference=preference)
    dfs_nodes = list(dict.fromkeys(sum(dfs_edges, ())))

    len_ori = {stream_tree.edges[x]['len'] for x in stream_tree_edge}
    bfs_prev = dict(nx.bfs_predecessors(stream_tree, source))
    bfs_next = dict(nx.bfs_successors(stream_tree, source))
    dict_tree = {}
    for x in dfs_nodes:
        dict_tree[x] = {'prev': "", 'next': []}
        if x in bfs_prev.keys():
            dict_tree[x]['prev'] = bfs_prev[x]
        if x in bfs_next.keys():
            x_rank = [dfs_nodes.index(x_next) for x_next in bfs_next[x]]
            dict_tree[x]['next'] = [
                y for _, y in sorted(
                    zip(x_rank, bfs_next[x]), key=lambda y: y[0])]

    dict_shift_dist = dict()  # shifting distance of each branch
    leaves = [n for n, d in stream_tree.degree() if d == 1]
    id_leaf = 0
    dfs_nodes_copy = deepcopy(dfs_nodes)
    num_nonroot_leaf = len(list(set(leaves) - set([source])))
    while len(dfs_nodes_copy) > 1:
        node = dfs_nodes_copy.pop()
        prev_node = dict_tree[node]['prev']
        if node in leaves:
            dict_shift_dist[(prev_node, node)] = \
                -(float(1)/dist_scale)*(num_nonroot_leaf-1)/2.0 \
                + id_leaf*(float(1)/dist_scale)
            id_leaf = id_leaf+1
        else:
            next_nodes = dict_tree[node]['next']
            dict_shift_dist[(prev_node, node)] = \
                (sum([dict_shift_dist[(node, next_node)]
                      for next_node in next_nodes]))/float(len(next_nodes))
    if stream_tree.degree(source) > 1:
        next_nodes = dict_tree[source]['next']
        dict_shift_dist[(source, source)] = \
            (sum([dict_shift_dist[(source, next_node)]
                  for next_node in next_nodes]))/float(len(next_nodes))

    # dataframe of bins
    df_bins = pd.DataFrame(
        index=['n_cells', 'boundary', 'center', 'edge'])
    dict_ann_df = {ann: pd.DataFrame(index=['n_cells'])
                   for ann in list_ann_numeric}
    # number of merged sliding windows
    dict_merge_num = {ann: [] for ann in list_ann_numeric}
    list_paths = _find_root_to_leaf_paths(stream_tree, source)
    max_path_len = _find_longest_path(list_paths, len_ori)
    size_w = max_path_len/float(factor_num_win)
    if size_w > min(len_ori.values())/float(factor_min_win):
        size_w = min(len_ori.values())/float(factor_min_win)
    # step of sliding window (the divisor should be an even number)
    step_w = size_w/2
    if len(dict_shift_dist) > 1:
        max_width = (max_path_len/float(factor_width))\
            / (max(dict_shift_dist.values())
                - min(dict_shift_dist.values()))
    else:
        max_width = max_path_len/float(factor_width)
    dict_shift_dist = {
        x: dict_shift_dist[x]*max_width for x in dict_shift_dist.keys()}
    min_width = 0.0  # min width of branch
    min_cellnum = 0  # the minimal cell number in one branch
    min_bin_cellnum = 0  # the minimal cell number in each bin
    # filter out cells whose total count
    # on one edge is below the min_cellnum
    dict_edge_filter = dict()
    df_edge_cellnum = pd.DataFrame(
        index=df_stream['label'].unique(),
        columns=dfs_edges,
        dtype=float)
    for i, edge_i in enumerate(dfs_edges):
        df_edge_i = df_stream[df_stream['st_edge'] == edge_i]
        cells_kept = df_edge_i['label'].value_counts()[
            df_edge_i['label'].value_counts() > min_cellnum].index
        df_edge_i = df_edge_i[df_edge_i['label'].isin(cells_kept)]
        dict_edge_filter[edge_i] = df_edge_i
        for cell_i in df_stream['label'].unique():
            df_edge_cellnum[edge_i][cell_i] = float(
                df_edge_i[df_edge_i['label'] == cell_i].shape[0])
    for i, edge_i in enumerate(dfs_edges):
        # degree of the start node
        degree_st = stream_tree.degree(edge_i[0])
        # degree of the end node
        degree_end = stream_tree.degree(edge_i[1])
        # matrix of windows only appearing on one edge
        mat_w = np.vstack([
            np.arange(0, len_ori[edge_i] - size_w + (
                len_ori[edge_i]/10**6), step_w),
            np.arange(size_w, len_ori[edge_i] + (
                len_ori[edge_i]/10**6), step_w)]).T
        mat_w[-1, -1] = len_ori[edge_i]
        if degree_st == 1:
            mat_w = np.insert(mat_w, 0, [0, size_w/2.0], axis=0)
        if degree_end == 1:
            mat_w = np.insert(
                mat_w, mat_w.shape[0],
                [len_ori[edge_i]-size_w/2.0, len_ori[edge_i]],
                axis=0)
        total_bins = df_bins.shape[1]  # current total number of bins

        if degree_st > 1 and i == 0:
            # avoid warning "DataFrame is highly fragmented."
            df_bins = df_bins.copy()
            # matrix of windows spanning multiple edges
            mat_w_common = np.array([[0, size_w/2.0], [0, size_w]])
            # neighbor nodes
            nb_nodes = list(stream_tree.neighbors(edge_i[0]))
            index_nb_nodes = [dfs_nodes.index(x) for x in nb_nodes]
            nb_nodes = \
                np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
            # matrix of windows spanning multiple edges
            total_bins = df_bins.shape[1]
            for i_win in range(mat_w_common.shape[0]):
                df_bins["win"+str(total_bins+i_win)] = ""
                df_bins.loc[df_bins.index[:-3],
                            "win"+str(total_bins+i_win)] = 0
                df_bins.at['edge', "win"+str(total_bins+i_win)] = \
                    [(source, source)]
                dict_df_ann_common = {ann: [] for ann in list_ann_numeric}
                for j in range(degree_st):
                    df_edge_j = dict_edge_filter[(edge_i[0], nb_nodes[j])]
                    cell_num_common2 = \
                        df_edge_j[np.logical_and(
                            df_edge_j['st_edge_loc'] >= 0,
                            df_edge_j['st_edge_loc'] <= mat_w_common[
                                i_win, 1])][ann].value_counts()
                    df_bins.loc[cell_num_common2.index,
                                "win"+str(total_bins + i_win)] = \
                        df_bins.loc[cell_num_common2.index,
                                    "win"+str(total_bins + i_win)]\
                        + cell_num_common2
                    for ann in list_ann_numeric:
                        dict_df_ann_common[ann].append(
                            df_edge_j[np.logical_and(
                                df_edge_j['st_edge_loc'] >= 0,
                                df_edge_j['st_edge_loc']
                                <= mat_w_common[i_win, 1])])
                    df_bins.loc[
                        'edge',
                        "win"+str(total_bins+i_win)].append(
                            (edge_i[0], nb_nodes[j]))
                for ann in list_ann_numeric:
                    ann_values_common = pd.concat(
                        dict_df_ann_common[ann]).groupby(['label'])[ann].mean()
                    dict_ann_df[ann].loc[
                        ann_values_common.index,
                        "win"+str(total_bins+i_win)] = ann_values_common
                    # avoid the warning "DataFrame is highly fragmented."
                    dict_ann_df[ann] = dict_ann_df[ann].copy()
                df_bins.at['boundary', "win"+str(total_bins+i_win)] = \
                    mat_w_common[i_win, :]
                if i_win == 0:
                    df_bins.loc['center', "win"+str(total_bins+i_win)] = 0
                else:
                    df_bins.loc['center',
                                "win"+str(total_bins+i_win)] = size_w/2

        # the maximal number of merging bins
        max_binnum = np.around((len_ori[edge_i]/4.0-size_w)/step_w)
        df_edge_i = dict_edge_filter[edge_i]
        total_bins = df_bins.shape[1]  # current total number of bins

        if max_binnum <= 1:
            df_bins = df_bins.copy()
            for i_win in range(mat_w.shape[0]):
                df_bins["win"+str(total_bins+i_win)] = ""
                df_bins.loc[df_bins.index[:-3],
                            "win"+str(total_bins+i_win)] = 0
                cell_num = df_edge_i[np.logical_and(
                    df_edge_i['st_edge_loc']
                    >= mat_w[i_win, 0],
                    df_edge_i['st_edge_loc']
                    <= mat_w[i_win, 1])][ann].value_counts()
                df_bins.loc[
                    cell_num.index,
                    "win"+str(total_bins+i_win)] = cell_num
                df_bins.at[
                    'boundary',
                    "win"+str(total_bins+i_win)] = mat_w[i_win, :]
                for ann in list_ann_numeric:
                    dict_ann_df[ann]["win"+str(total_bins+i_win)] = 0
                    ann_values = df_edge_i[np.logical_and(
                        df_edge_i['st_edge_loc'] >= mat_w[i_win, 0],
                        df_edge_i['st_edge_loc'] <= mat_w[i_win, 1]
                        )].groupby(['label'])[ann].mean()
                    dict_ann_df[ann].loc[
                        ann_values.index,
                        "win"+str(total_bins+i_win)] = ann_values
                    # avoid warning "DataFrame is highly fragmented."
                    dict_ann_df[ann] = dict_ann_df[ann].copy()
                    dict_merge_num[ann].append(1)
                if degree_st == 1 and i_win == 0:
                    df_bins.loc['center', "win"+str(total_bins+i_win)] = 0
                elif degree_end == 1 and i_win == (mat_w.shape[0]-1):
                    df_bins.loc[
                        'center',
                        "win"+str(total_bins+i_win)] = len_ori[edge_i]
                else:
                    df_bins.loc[
                        'center',
                        "win"+str(total_bins+i_win)] =\
                            np.mean(mat_w[i_win, :])
            col_wins = [
                "win"+str(total_bins+i_win)
                for i_win in range(mat_w.shape[0])]
            df_bins.loc['edge', col_wins] = pd.Series(
                index=col_wins,
                data=[[edge_i]]*len(col_wins))

        if max_binnum > 1:
            df_bins = df_bins.copy()
            id_stack = []
            for i_win in range(mat_w.shape[0]):
                id_stack.append(i_win)
                bd_bins = [
                    mat_w[id_stack[0], 0],
                    mat_w[id_stack[-1], 1]]  # boundary of merged bins
                cell_num = df_edge_i[np.logical_and(
                    df_edge_i['st_edge_loc'] >=
                    bd_bins[0],
                    df_edge_i['st_edge_loc'] <=
                    bd_bins[1])][ann].value_counts()
                if len(id_stack) == max_binnum \
                        or any(cell_num > min_bin_cellnum) \
                        or i_win == mat_w.shape[0]-1:
                    df_bins["win"+str(total_bins)] = ""
                    df_bins.loc[
                        df_bins.index[:-3],
                        "win"+str(total_bins)] = 0
                    df_bins.loc[
                        cell_num.index,
                        "win"+str(total_bins)] = cell_num
                    df_bins.at['boundary', "win"+str(total_bins)] = bd_bins
                    df_bins.at['edge', "win"+str(total_bins)] = [edge_i]
                    for ann in list_ann_numeric:
                        dict_ann_df[ann]["win"+str(total_bins)] = 0
                        ann_values = df_edge_i[np.logical_and(
                            df_edge_i['st_edge_loc'] >= bd_bins[0],
                            df_edge_i['st_edge_loc'] <= bd_bins[1]
                            )].groupby(['CELL_LABEL'])[ann].mean()
                        dict_ann_df[ann].loc[
                            ann_values.index,
                            "win"+str(total_bins)] = ann_values
                        dict_ann_df[ann] = dict_ann_df[ann].copy()
                        dict_merge_num[ann].append(len(id_stack))
                    if degree_st == 1 and (0 in id_stack):
                        df_bins.loc['center', "win"+str(total_bins)] = 0
                    elif degree_end == 1 and i_win == (mat_w.shape[0]-1):
                        df_bins.loc[
                            'center',
                            "win"+str(total_bins)] = len_ori[edge_i]
                    else:
                        df_bins.loc[
                            'center',
                            "win"+str(total_bins)] = np.mean(bd_bins)
                    total_bins = total_bins + 1
                    id_stack = []

        if degree_end > 1:
            df_bins = df_bins.copy()
            # matrix of windows appearing on multiple edges
            mat_w_common = np.vstack(
                [np.arange(
                    len_ori[edge_i] - size_w + step_w,
                    len_ori[edge_i] + (len_ori[edge_i]/10**6),
                    step_w),
                    np.arange(
                    step_w,
                    size_w+(len_ori[edge_i]/10**6),
                    step_w)]).T
            # neighbor nodes
            nb_nodes = list(stream_tree.neighbors(edge_i[1]))
            nb_nodes.remove(edge_i[0])
            index_nb_nodes = [dfs_nodes.index(x) for x in nb_nodes]
            nb_nodes = np.array(nb_nodes)[
                np.argsort(index_nb_nodes)].tolist()

            # matrix of windows appearing on multiple edges
            total_bins = df_bins.shape[1]  # current total number of bins
            if mat_w_common.shape[0] > 0:
                for i_win in range(mat_w_common.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[
                        df_bins.index[:-3],
                        "win"+str(total_bins+i_win)] = 0
                    cell_num_common1 =\
                        df_edge_i[np.logical_and(
                            df_edge_i['st_edge_loc']
                            > mat_w_common[i_win, 0],
                            df_edge_i['st_edge_loc']
                            <= len_ori[edge_i])][ann].value_counts()
                    df_bins.loc[
                        cell_num_common1.index,
                        "win"+str(total_bins+i_win)] = cell_num_common1
                    dict_df_ann_common = dict()
                    for ann in list_ann_numeric:
                        dict_ann_df[ann]["win"+str(total_bins+i_win)] = 0
                        dict_df_ann_common[ann] = list()
                        dict_df_ann_common[ann].append(
                            df_edge_i[np.logical_and(
                                df_edge_i['st_edge_loc']
                                > mat_w_common[i_win, 0],
                                df_edge_i['st_edge_loc']
                                <= len_ori[edge_i])])
                        dict_merge_num[ann].append(1)
                    df_bins.at[
                        'edge', "win"+str(total_bins+i_win)] = [edge_i]
                    for j in range(degree_end - 1):
                        df_edge_j = dict_edge_filter[(
                            edge_i[1], nb_nodes[j])]
                        cell_num_common2 = \
                            df_edge_j[np.logical_and(
                                df_edge_j['st_edge_loc'] >= 0,
                                df_edge_j['st_edge_loc']
                                <= mat_w_common[i_win, 1]
                                )][ann].value_counts()
                        df_bins.loc[cell_num_common2.index,
                                    "win"+str(total_bins+i_win)] = \
                            df_bins.loc[
                                cell_num_common2.index,
                                "win"+str(total_bins+i_win)] \
                            + cell_num_common2
                        for ann in list_ann_numeric:
                            dict_df_ann_common[ann].append(
                                df_edge_j[np.logical_and(
                                    df_edge_j['st_edge_loc']
                                    >= 0,
                                    df_edge_j['st_edge_loc']
                                    <= mat_w_common[i_win, 1])])
                        if abs(((sum(
                            mat_w_common[i_win, :])
                            + len_ori[edge_i])/2)
                            - (len_ori[edge_i]
                                + size_w/2.0)) < step_w/100.0:
                            df_bins.loc[
                                'edge',
                                "win"+str(total_bins+i_win)].append(
                                    (edge_i[1], nb_nodes[j]))
                    for ann in list_ann_numeric:
                        ann_values_common = pd.concat(
                            dict_df_ann_common[ann]).groupby(
                                ['label'])[ann].mean()
                        dict_ann_df[ann].loc[
                            ann_values_common.index,
                            "win"+str(total_bins + i_win)] = ann_values_common
                        dict_ann_df[ann] = dict_ann_df[ann].copy()
                    df_bins.at['boundary', "win"+str(total_bins+i_win)] = \
                        mat_w_common[i_win, :]
                    df_bins.loc['center', "win"+str(total_bins+i_win)] = \
                        (sum(mat_w_common[i_win, :])+len_ori[edge_i])/2

    df_bins = df_bins.copy()
    # order cell names by the index of first non-zero
    cell_list = df_bins.index[:-3]
    id_nonzero = []
    for cellname in cell_list:
        if np.flatnonzero(
                df_bins.loc[cellname, ]).size == 0:
            print('Cell '+cellname+' does not exist')
            break
        else:
            id_nonzero.append(
                np.flatnonzero(df_bins.loc[cellname, ])[0])
    cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()
    # original count
    df_bins_ori = df_bins.reindex(
        cell_list_sorted+['boundary', 'center', 'edge'])
    if log_scale:
        df_n_cells = df_bins_ori.iloc[:-3, :].sum()
        df_n_cells = df_n_cells/df_n_cells.max()*factor_zoomin
        df_bins_ori.iloc[:-3, :] = \
            df_bins_ori.iloc[:-3, :]*np.log2(df_n_cells+1)/(df_n_cells+1)
    df_bins_cumsum = df_bins_ori.copy()
    df_bins_cumsum.iloc[:-3, :] = \
        df_bins_ori.iloc[:-3, :][::-1].cumsum()[::-1]

    # normalization
    df_bins_cumsum_norm = df_bins_cumsum.copy()
    df_bins_cumsum_norm.iloc[:-3, :] = \
        min_width + max_width*(
            df_bins_cumsum.iloc[:-3, :])\
        / (df_bins_cumsum.iloc[:-3, :]).values.max()

    df_bins_top = df_bins_cumsum_norm.copy()
    df_bins_top.iloc[:-3, :] = \
        df_bins_cumsum_norm.iloc[:-3, :].subtract(
            df_bins_cumsum_norm.iloc[0, :]/2.0)
    df_bins_base = df_bins_top.copy()
    df_bins_base.iloc[:-4, :] = df_bins_top.iloc[1:-3, ].values
    df_bins_base.iloc[-4, :] = 0-df_bins_cumsum_norm.iloc[0, :]/2.0
    dict_forest = {
        cellname: {nodename: {'prev': "", 'next': "", 'div': ""}
                   for nodename in dfs_nodes}
        for cellname in df_edge_cellnum.index}
    for cellname in cell_list_sorted:
        for node_i in dfs_nodes:
            nb_nodes = list(stream_tree.neighbors(node_i))
            index_in_bfs = [dfs_nodes.index(nb) for nb in nb_nodes]
            nb_nodes_sorted = np.array(
                nb_nodes)[np.argsort(index_in_bfs)].tolist()
            if node_i == source:
                next_nodes = nb_nodes_sorted
                prev_nodes = ''
            else:
                next_nodes = nb_nodes_sorted[1:]
                prev_nodes = nb_nodes_sorted[0]
            dict_forest[cellname][node_i]['next'] = next_nodes
            dict_forest[cellname][node_i]['prev'] = prev_nodes
            if len(next_nodes) > 1:
                pro_next_edges = []  # proportion of next edges
                for nt in next_nodes:
                    id_wins = [ix for ix, x in enumerate(
                        df_bins_cumsum_norm.loc['edge', :])
                        if x == [(node_i, nt)]]
                    pro_next_edges.append(
                        df_bins_cumsum_norm.loc[
                            cellname, 'win'+str(id_wins[0])])
                if sum(pro_next_edges) == 0:
                    dict_forest[cellname][node_i]['div'] = \
                        np.cumsum(np.repeat(1.0/len(next_nodes),
                                  len(next_nodes))).tolist()
                else:
                    dict_forest[cellname][node_i]['div'] = \
                        (np.cumsum(
                            pro_next_edges)/sum(
                                pro_next_edges)).tolist()

    # Shift
    # coordinates of end points
    dict_ep_top = {cellname: dict() for cellname in cell_list_sorted}
    dict_ep_base = {cellname: dict() for cellname in cell_list_sorted}
    # center coordinates of end points in each branch
    dict_ep_center = dict()

    df_top_x = df_bins_top.copy()  # x coordinates in top line
    df_top_y = df_bins_top.copy()  # y coordinates in top line
    df_base_x = df_bins_base.copy()  # x coordinates in base line
    df_base_y = df_bins_base.copy()  # y coordinates in base line

    for edge_i in dfs_edges:
        id_wins = [i for i, x in enumerate(
            df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
        prev_node = dict_tree[edge_i[0]]['prev']
        if prev_node == '':
            x_st = 0
            if stream_tree.degree(source) > 1:
                id_wins = id_wins[1:]
        else:
            id_wins = id_wins[1:]  # remove the overlapped window
            x_st = dict_ep_center[(prev_node, edge_i[0])][0] - step_w
        y_st = dict_shift_dist[edge_i]
        for cellname in cell_list_sorted:
            # top line
            px_top = df_bins_top.loc[
                'center', list(map(lambda x: 'win' + str(x), id_wins))]
            py_top = df_bins_top.loc[
                cellname, list(map(lambda x: 'win' + str(x), id_wins))]
            px_top_prime = x_st + px_top
            py_top_prime = y_st + py_top
            dict_ep_top[cellname][edge_i] = \
                [px_top_prime[-1], py_top_prime[-1]]
            df_top_x.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins))]\
                = px_top_prime
            df_top_y.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins))]\
                = py_top_prime
            # base line
            px_base = df_bins_base.loc[
                'center',
                list(map(lambda x: 'win' + str(x), id_wins))]
            py_base = df_bins_base.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins))]
            px_base_prime = x_st + px_base
            py_base_prime = y_st + py_base
            dict_ep_base[cellname][edge_i] = \
                [px_base_prime[-1], py_base_prime[-1]]
            df_base_x.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins))]\
                = px_base_prime
            df_base_y.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins))]\
                = py_base_prime
        dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

    id_wins_start = [i for i, x in enumerate(
        df_bins_cumsum_norm.loc['edge', :])
        if x[0] == (source, source)]
    if len(id_wins_start) > 0:
        mean_shift_dist =\
            np.mean([dict_shift_dist[(source, x)]
                    for x in dict_forest[
                        cell_list_sorted[0]][source]['next']])
        for cellname in cell_list_sorted:
            # top line
            px_top = df_bins_top.loc[
                'center',
                list(map(lambda x: 'win' + str(x), id_wins_start))]
            py_top = df_bins_top.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins_start))]
            px_top_prime = 0 + px_top
            py_top_prime = mean_shift_dist + py_top
            df_top_x.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins_start))]\
                = px_top_prime
            df_top_y.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins_start))]\
                = py_top_prime
            # base line
            px_base = df_bins_base.loc[
                'center',
                list(map(lambda x: 'win' + str(x), id_wins_start))]
            py_base = df_bins_base.loc[
                cellname,
                list(map(lambda x: 'win' + str(x), id_wins_start))]
            px_base_prime = 0 + px_base
            py_base_prime = mean_shift_dist + py_base
            df_base_x.loc[
                cellname,
                list(map(lambda x: 'win' + str(x),
                         id_wins_start))] = px_base_prime
            df_base_y.loc[
                cellname,
                list(map(lambda x: 'win' + str(x),
                         id_wins_start))] = py_base_prime

    # determine joints points
    # coordinates of joint points
    dict_joint_top = {
        cellname: dict() for cellname in cell_list_sorted}
    dict_joint_base = {
        cellname: dict() for cellname in cell_list_sorted}
    if stream_tree.degree(source) == 1:
        id_joints = [i for i, x in enumerate(
            df_bins_cumsum_norm.loc['edge', :]) if len(x) > 1]
    else:
        id_joints = [i for i, x in enumerate(
            df_bins_cumsum_norm.loc['edge', :])
            if len(x) > 1 and x[0] != (source, source)]
        id_joints.insert(0, 1)
    for id_j in id_joints:
        joint_edges = df_bins_cumsum_norm.loc['edge', 'win'+str(id_j)]
        for id_div, edge_i in enumerate(joint_edges[1:]):
            id_wins = [i for i, x in enumerate(
                df_bins_cumsum_norm.loc['edge', :]) if x == [edge_i]]
            for cellname in cell_list_sorted:
                if len(dict_forest[cellname][edge_i[0]]['div']) > 0:
                    prev_node_top_x = df_top_x.loc[
                        cellname, 'win'+str(id_j)]
                    prev_node_top_y = df_top_y.loc[
                        cellname, 'win'+str(id_j)]
                    prev_node_base_x = df_base_x.loc[
                        cellname, 'win'+str(id_j)]
                    prev_node_base_y = df_base_y.loc[
                        cellname, 'win'+str(id_j)]
                    div = dict_forest[cellname][edge_i[0]]['div']
                    if id_div == 0:
                        px_top_prime_st = prev_node_top_x
                        py_top_prime_st = prev_node_top_y
                    else:
                        px_top_prime_st = prev_node_top_x \
                            + (prev_node_base_x - prev_node_top_x)\
                            * div[id_div-1]
                        py_top_prime_st = prev_node_top_y \
                            + (prev_node_base_y - prev_node_top_y)\
                            * div[id_div-1]
                    px_base_prime_st = prev_node_top_x \
                        + (prev_node_base_x - prev_node_top_x)*div[id_div]
                    py_base_prime_st = prev_node_top_y \
                        + (prev_node_base_y - prev_node_top_y)*div[id_div]
                    df_top_x.loc[cellname, 'win'+str(
                        id_wins[0])] = px_top_prime_st
                    df_top_y.loc[cellname, 'win'+str(
                        id_wins[0])] = py_top_prime_st
                    df_base_x.loc[cellname, 'win'+str(
                        id_wins[0])] = px_base_prime_st
                    df_base_y.loc[cellname, 'win'+str(
                        id_wins[0])] = py_base_prime_st
                    dict_joint_top[cellname][edge_i] =\
                        np.array([px_top_prime_st, py_top_prime_st])
                    dict_joint_base[cellname][edge_i] =\
                        np.array([px_base_prime_st, py_base_prime_st])

    dict_tree_copy = deepcopy(dict_tree)
    dict_paths_top, dict_paths_base = \
        _find_paths(dict_tree_copy, dfs_nodes)

    # identify boundary of each edge
    dict_edge_bd = dict()
    for edge_i in dfs_edges:
        id_wins = [i for i, x in enumerate(
            df_top_x.loc['edge', :]) if edge_i in x]
        dict_edge_bd[edge_i] = [
            df_top_x.iloc[0, id_wins[0]], df_top_x.iloc[0, id_wins[-1]]]

    x_smooth = np.unique(np.arange(
        min(df_base_x.iloc[0, :]),
        max(df_base_x.iloc[0, :]),
        step=step_w/20).tolist() + [max(df_base_x.iloc[0, :])]).tolist()
    x_joints = df_top_x.iloc[0, id_joints].tolist()
    # replace nearest value in x_smooth by x axis of joint points
    for x in x_joints:
        x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

    dict_smooth_linear = {cellname: {
        'top': dict(), 'base': dict()} for cellname in cell_list_sorted}
    # interpolation
    for edge_i_top in dict_paths_top.keys():
        path_i_top = dict_paths_top[edge_i_top]
        id_wins_top = [i_x for i_x, x in enumerate(
            df_top_x.loc['edge']) if set(
                np.unique(x)).issubset(set(path_i_top))]
        if stream_tree.degree(source) > 1 \
            and edge_i_top == (
                source, dict_forest[
                    cell_list_sorted[0]][source]['next'][0]):
            id_wins_top.insert(0, 1)
            id_wins_top.insert(0, 0)
        for cellname in cell_list_sorted:
            x_top = df_top_x.loc[cellname, list(map(
                lambda x: 'win' + str(x), id_wins_top))].tolist()
            y_top = df_top_y.loc[cellname, list(map(
                lambda x: 'win' + str(x), id_wins_top))].tolist()
            f_top_linear = interpolate.interp1d(
                x_top, y_top, kind='linear')
            x_top_new = [x for x in x_smooth if (
                x >= x_top[0]) and (x <= x_top[-1])] + [x_top[-1]]
            x_top_new = np.unique(x_top_new).tolist()
            y_top_new_linear = f_top_linear(x_top_new)
            for id_node in range(len(path_i_top)-1):
                edge_i = (path_i_top[id_node], path_i_top[id_node+1])
                edge_i_bd = dict_edge_bd[edge_i]
                id_selected = [i_x for i_x, x in enumerate(
                    x_top_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                dict_smooth_linear[cellname]['top'][edge_i] = \
                    pd.DataFrame([
                        np.array(x_top_new)[id_selected],
                        np.array(y_top_new_linear)[id_selected]],
                                    index=['x', 'y'])
    for edge_i_base in dict_paths_base.keys():
        path_i_base = dict_paths_base[edge_i_base]
        id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge'])
                        if set(np.unique(x)).issubset(set(path_i_base))]
        if stream_tree.degree(source) > 1 \
            and edge_i_base == (source, dict_forest[
                cell_list_sorted[0]][source]['next'][-1]):
            id_wins_base.insert(0, 1)
            id_wins_base.insert(0, 0)
        for cellname in cell_list_sorted:
            x_base = df_base_x.loc[cellname, list(map(
                lambda x: 'win' + str(x), id_wins_base))].tolist()
            y_base = df_base_y.loc[cellname, list(map(
                lambda x: 'win' + str(x), id_wins_base))].tolist()
            f_base_linear = interpolate.interp1d(
                x_base, y_base, kind='linear')
            x_base_new = [x for x in x_smooth if (x >= x_base[0]) and (
                x <= x_base[-1])] + [x_base[-1]]
            x_base_new = np.unique(x_base_new).tolist()
            y_base_new_linear = f_base_linear(x_base_new)
            for id_node in range(len(path_i_base)-1):
                edge_i = (path_i_base[id_node], path_i_base[id_node+1])
                edge_i_bd = dict_edge_bd[edge_i]
                id_selected = [i_x for i_x, x in enumerate(
                    x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                dict_smooth_linear[cellname]['base'][edge_i] = \
                    pd.DataFrame([
                        np.array(x_base_new)[id_selected],
                        np.array(y_base_new_linear)[id_selected]],
                        index=['x', 'y'])

    # searching for edges on which cell exists
    # based on the linear interpolation
    dict_edges_CE = {cellname: [] for cellname in cell_list_sorted}
    for cellname in cell_list_sorted:
        for edge_i in dfs_edges:
            if sum(abs(
                    dict_smooth_linear[
                        cellname]['top'][edge_i].loc['y']
                    - dict_smooth_linear[
                        cellname]['base'][edge_i].loc['y'])
                    > 1e-12):
                dict_edges_CE[cellname].append(edge_i)

    # determine paths where cell exists
    dict_paths_CE_top = {cellname: {} for cellname in cell_list_sorted}
    dict_paths_CE_base = {cellname: {} for cellname in cell_list_sorted}
    dict_forest_CE = dict()
    for cellname in cell_list_sorted:
        edges_cn = dict_edges_CE[cellname]
        nodes = [nodename for nodename in dfs_nodes if nodename in set(
            itertools.chain(*edges_cn))]
        dict_forest_CE[cellname] = {
            nodename: {'prev': "", 'next': []} for nodename in nodes}
        for node_i in nodes:
            prev_node = dict_tree[node_i]['prev']
            if (prev_node, node_i) in edges_cn:
                dict_forest_CE[cellname][node_i]['prev'] = prev_node
            next_nodes = dict_tree[node_i]['next']
            for x in next_nodes:
                if (node_i, x) in edges_cn:
                    (dict_forest_CE[cellname][node_i]['next']).append(x)
        dict_paths_CE_top[cellname], dict_paths_CE_base[cellname] = \
            _find_paths(dict_forest_CE[cellname], nodes)

    dict_smooth_new = deepcopy(dict_smooth_linear)
    for cellname in cell_list_sorted:
        paths_CE_top = dict_paths_CE_top[cellname]
        for edge_i_top in paths_CE_top.keys():
            path_i_top = paths_CE_top[edge_i_top]
            edges_top = [x for x in dfs_edges if set(
                np.unique(x)).issubset(set(path_i_top))]
            id_wins_top = [i_x for i_x, x in enumerate(
                df_top_x.loc['edge']) if set(
                    np.unique(x)).issubset(set(path_i_top))]

            x_top = []
            y_top = []
            for e_t in edges_top:
                if e_t == edges_top[-1]:
                    py_top_linear = dict_smooth_linear[
                        cellname]['top'][e_t].loc['y']
                    px = dict_smooth_linear[
                        cellname]['top'][e_t].loc['x']
                else:
                    py_top_linear = dict_smooth_linear[
                        cellname]['top'][e_t].iloc[1, :-1]
                    px = dict_smooth_linear[
                        cellname]['top'][e_t].iloc[0, :-1]
                x_top = x_top + px.tolist()
                y_top = y_top + py_top_linear.tolist()
            x_top_new = x_top
            y_top_new = savgol_filter(y_top, 11, polyorder=1)
            for id_node in range(len(path_i_top)-1):
                edge_i = (path_i_top[id_node], path_i_top[id_node+1])
                edge_i_bd = dict_edge_bd[edge_i]
                id_selected = [i_x for i_x, x in enumerate(
                    x_top_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                dict_smooth_new[cellname]['top'][edge_i] = \
                    pd.DataFrame([
                        np.array(x_top_new)[id_selected],
                        np.array(y_top_new)[id_selected]],
                        index=['x', 'y'])
        paths_CE_base = dict_paths_CE_base[cellname]
        for edge_i_base in paths_CE_base.keys():
            path_i_base = paths_CE_base[edge_i_base]
            edges_base = [x for x in dfs_edges if set(
                np.unique(x)).issubset(set(path_i_base))]
            id_wins_base = [i_x for i_x, x in enumerate(
                df_base_x.loc['edge'])
                if set(np.unique(x)).issubset(set(path_i_base))]

            x_base = []
            y_base = []
            for e_b in edges_base:
                if e_b == edges_base[-1]:
                    py_base_linear = dict_smooth_linear[
                        cellname]['base'][e_b].loc['y']
                    px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                else:
                    py_base_linear = dict_smooth_linear[
                        cellname]['base'][e_b].iloc[1, :-1]
                    px = dict_smooth_linear[
                        cellname]['base'][e_b].iloc[0, :-1]
                x_base = x_base + px.tolist()
                y_base = y_base + py_base_linear.tolist()
            x_base_new = x_base
            y_base_new = savgol_filter(y_base, 11, polyorder=1)
            for id_node in range(len(path_i_base)-1):
                edge_i = (path_i_base[id_node], path_i_base[id_node+1])
                edge_i_bd = dict_edge_bd[edge_i]
                id_selected = [i_x for i_x, x in enumerate(
                    x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                dict_smooth_new[cellname]['base'][edge_i] = \
                    pd.DataFrame([
                        np.array(x_base_new)[id_selected],
                        np.array(y_base_new)[id_selected]],
                        index=['x', 'y'])
    # find all edges of polygon
    poly_edges = []
    dict_tree_copy = deepcopy(dict_tree)
    cur_node = source
    next_node = dict_tree_copy[cur_node]['next'][0]
    dict_tree_copy[cur_node]['next'].pop(0)
    poly_edges.append((cur_node, next_node))
    cur_node = next_node
    while not ((next_node == source) and (
            cur_node == dict_tree[source]['next'][-1])):
        while len(dict_tree_copy[cur_node]['next']) != 0:
            next_node = dict_tree_copy[cur_node]['next'][0]
            dict_tree_copy[cur_node]['next'].pop(0)
            poly_edges.append((cur_node, next_node))
            if cur_node == dict_tree[source]['next'][-1] \
                    and next_node == source:
                break
            cur_node = next_node
        while len(dict_tree_copy[cur_node]['next']) == 0:
            next_node = dict_tree_copy[cur_node]['prev']
            poly_edges.append((cur_node, next_node))
            if cur_node == dict_tree[source]['next'][-1] \
                    and next_node == source:
                break
            cur_node = next_node

    verts = {cellname: np.empty((0, 2)) for cellname in cell_list_sorted}
    for cellname in cell_list_sorted:
        for edge_i in poly_edges:
            if edge_i in dfs_edges:
                x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                pxy = np.array([x_top, y_top]).T
            else:
                edge_i = (edge_i[1], edge_i[0])
                x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                x_base = x_base[::-1]
                y_base = y_base[::-1]
                pxy = np.array([x_base, y_base]).T
            verts[cellname] = np.vstack((verts[cellname], pxy))
    dict_verts[ann] = verts

    extent = {'xmin': "", 'xmax': "", 'ymin': "", 'ymax': ""}
    for cellname in cell_list_sorted:
        for edge_i in dfs_edges:
            xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
            xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
            ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
            ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
            if extent['xmin'] == "":
                extent['xmin'] = xmin
            else:
                if xmin < extent['xmin']:
                    extent['xmin'] = xmin

            if extent['xmax'] == "":
                extent['xmax'] = xmax
            else:
                if xmax > extent['xmax']:
                    extent['xmax'] = xmax

            if extent['ymin'] == "":
                extent['ymin'] = ymin
            else:
                if ymin < extent['ymin']:
                    extent['ymin'] = ymin

            if extent['ymax'] == "":
                extent['ymax'] = ymax
            else:
                if ymax > extent['ymax']:
                    extent['ymax'] = ymax

    dict_im_array = dict()
    for ann in list_ann_numeric:
        im_nrow = factor_nrow
        im_ncol = factor_ncol
        xmin = extent['xmin']
        xmax = extent['xmax']
        ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
        ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1
        im_array = {cellname: np.zeros((im_nrow, im_ncol))
                    for cellname in cell_list_sorted}
        df_bins_ann = dict_ann_df[ann]
        for cellname in cell_list_sorted:
            for edge_i in dfs_edges:
                id_wins_all = [i for i, x in enumerate(
                    df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
                prev_edge = ''
                id_wins_prev = []
                if stream_tree.degree(source) > 1:
                    if edge_i == dfs_edges[0]:
                        id_wins = [0, 1]
                        im_array = _fill_im_array(
                            im_array, df_bins_ann, stream_tree,
                            df_base_x, df_base_y, df_top_y,
                            xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                            dict_shift_dist, id_wins, edge_i, cellname,
                            id_wins_prev, prev_edge)
                    id_wins = id_wins_all
                    if edge_i[0] == source:
                        prev_edge = (source, source)
                        id_wins_prev = [0, 1]
                    else:
                        prev_edge = (dict_tree[edge_i[0]]['prev'], edge_i[0])
                        id_wins_prev = [i for i, x in enumerate(
                            df_bins_cumsum_norm.loc['edge', :])
                            if x[0] == prev_edge]
                    im_array = _fill_im_array(
                        im_array, df_bins_ann, stream_tree,
                        df_base_x, df_base_y, df_top_y,
                        xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                        dict_shift_dist, id_wins, edge_i, cellname,
                        id_wins_prev, prev_edge)
                else:
                    id_wins = id_wins_all
                    if edge_i[0] != source:
                        prev_edge = (dict_tree[edge_i[0]]['prev'], edge_i[0])
                        id_wins_prev = [i for i, x in enumerate(
                            df_bins_cumsum_norm.loc['edge', :])
                            if x[0] == prev_edge]
                    im_array = _fill_im_array(
                        im_array, df_bins_ann, stream_tree,
                        df_base_x, df_base_y, df_top_y,
                        xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                        dict_shift_dist, id_wins, edge_i, cellname,
                        id_wins_prev, prev_edge)
        dict_im_array[ann] = im_array
    return verts, extent, cell_list_sorted, dict_ann_df, dict_im_array
