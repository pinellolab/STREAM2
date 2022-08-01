"""Utility functions and classes."""

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
import itertools


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


# def add_stream_sc_pos(
#         adata,
#         source=0,
#         dist_scale=1,
#         dist_pctl=95,
#         preference=None
# ):

#     epg_edge = adata.uns['epg']['edge']
#     stream_tree = nx.Graph()
#     stream_tree_edge = adata.uns['stream_tree']["edge"]
#     stream_tree_edge_len = adata.uns['stream_tree']["edge_len"]
#     edges_weighted = list(zip(
#         stream_tree_edge[:, 0],
#         stream_tree_edge[:, 1],
#         stream_tree_edge_len))
#     stream_tree.add_weighted_edges_from(edges_weighted)

#     dict_bfs_pre = dict(nx.bfs_predecessors(stream_tree, source))
#     dict_bfs_suc = dict(nx.bfs_successors(stream_tree, source))
#     dict_edge_shift_dist = \
#         _calculate_shift_distance(
#             adata, source=source,
#             dist_pctl=dist_pctl, preference=preference)
#     dict_path_len = nx.shortest_path_length(
#         stream_tree, source=source, weight='len')
#     df_cells_pos = pd.DataFrame(index=adata.obs.index, columns=['cells_pos'])
#     dict_edge_pos = {}
#     dict_node_pos = {}
#     for edge in dict_edge_shift_dist.keys():
#         node_pos_st = np.array(
#             [dict_path_len[edge[0]], dict_edge_shift_dist[edge]])
#         node_pos_ed = np.array(
#             [dict_path_len[edge[1]], dict_edge_shift_dist[edge]])  
#         br_id = stream_tree.edges[edge]['id']
#         id_cells = np.where(adata.obs['branch_id']==br_id)[0]
#         # cells_pos_x = flat_tree.nodes[root_node]['pseudotime'].iloc[id_cells]
#         cells_pos_x = adata.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
#         np.random.seed(100)
#         cells_pos_y = node_pos_st[1] + dist_scale*adata.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
#         cells_pos = np.array((cells_pos_x,cells_pos_y)).T
#         df_cells_pos.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
#         dict_edge_pos[edge] = np.array([node_pos_st,node_pos_ed])    
#         if(edge[0] not in dict_bfs_pre.keys()):
#             dict_node_pos[edge[0]] = node_pos_st
#         dict_node_pos[edge[1]] = node_pos_ed
#     adata.obsm['X_stream_'+root] = np.array(df_cells_pos['cells_pos'].tolist())
    
#     if(flat_tree.degree(root_node)>1):
#         suc_nodes = dict_bfs_suc[root_node]
#         edges = [(root_node,sn) for sn in suc_nodes]
#         max_y_pos = max([dict_edge_pos[x][0,1] for x in edges])
#         min_y_pos = min([dict_edge_pos[x][0,1] for x in edges])
#         median_y_pos = np.median([dict_edge_pos[x][0,1] for x in edges])
#         x_pos = dict_edge_pos[edges[0]][0,0]
#         dict_node_pos[root_node] = np.array([x_pos,median_y_pos])

#     adata.uns['stream_'+root] = dict()
#     adata.uns['stream_'+root]['nodes'] = dict_node_pos
#     adata.uns['stream_'+root]['edges'] = dict()
     
#     for edge in dict_edge_pos.keys():  
#         edge_pos = dict_edge_pos[edge]
#         edge_color = flat_tree.edges[edge]['color']
#         if(edge[0] in dict_bfs_pre.keys()):
#             pre_node = dict_bfs_pre[edge[0]]
#             link_edge_pos = np.array([dict_edge_pos[(pre_node,edge[0])][1,],dict_edge_pos[edge][0,]])
#             edge_pos = np.vstack((link_edge_pos,edge_pos))
#         adata.uns['stream_'+root]['edges'][edge]=edge_pos
#     if(flat_tree.degree(root_node)>1):
#         suc_nodes = dict_bfs_suc[root_node]
#         edges = [(root_node,sn) for sn in suc_nodes]
#         max_y_pos = max([dict_edge_pos[x][0,1] for x in edges])
#         min_y_pos = min([dict_edge_pos[x][0,1] for x in edges])
#         x_pos = dict_node_pos[root_node][0]
#         link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
#         adata.uns['stream_'+root]['edges'][(root_node,root_node)]=link_edge_pos