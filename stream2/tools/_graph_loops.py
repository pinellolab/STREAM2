import networkx as nx
import elpigraph
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scanpy as sc
import itertools

from sklearn.decomposition import PCA
from scipy.spatial.qhull import _Qhull
from shapely.geometry import Point, Polygon, MultiLineString, LineString
from shapely.geometry.multipolygon import MultiPolygon
from sklearn.neighbors import NearestNeighbors

from ._elpigraph import learn_graph
from ._graph_editing import _get_graph_data


@nb.njit
def _get_intersect_inner(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack((a1, a2, b1, b2))  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return np.array([np.inf, np.inf])
    return np.array([x / z, y / z])


@nb.njit
def _isBetween(a, b, c):
    """ Check if c is in between a and b """
    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False
    return True


@nb.njit
def _get_intersect(Xin, nodep, edges, cent):
    inters = np.zeros_like(Xin)
    for i, x in enumerate(Xin):
        for e in edges:
            p_inter = _get_intersect_inner(nodep[e[0]], nodep[e[1]], x, cent)
            if _isBetween(nodep[e[0]], nodep[e[1]], p_inter):
                if np.sum((x - p_inter) ** 2) < np.sum((cent - p_inter) ** 2):
                    inters[i] = p_inter
    return inters


@nb.njit
def get_weights_lineproj(Xin, nodep, edges, cent, threshold=0.2):

    Xin_lineproj = _get_intersect(Xin, nodep, edges, cent)
    distcent_Xin_lineproj = np.sqrt(np.sum((Xin_lineproj - cent) ** 2, axis=1))
    distcent_Xin = np.sqrt(np.sum((Xin - cent) ** 2, axis=1))

    w = 1 - distcent_Xin / distcent_Xin_lineproj
    idx_close = w > threshold
    w[idx_close] = 1.0
    return w, idx_close


def shrink_or_swell_shapely_polygon(coords, factor=0.10, swell=False):
    """returns the shapely polygon which is smaller or bigger by passed factor.
    If swell = True , then it returns bigger polygon, else smaller"""

    my_polygon = Polygon(coords)
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * factor

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

    return my_polygon_resized


def remove_intersections(nodep, edges):
    """ Update edges to account for possible 2d intersections in graph after adding loops """
    new_nodep = nodep.copy()
    lnodep = new_nodep.tolist()
    new_edges = edges.tolist()
    multiline = MultiLineString([LineString(new_nodep[e]) for e in new_edges])

    while not (multiline.is_simple):  # while intersections in graph

        # find an intersection, update edges, break, update graph
        for i, j in itertools.combinations(range(len(multiline)), 2):
            line1, line2 = multiline[i], multiline[j]
            if line1.intersects(line2):
                if list(np.array(line1.intersection(line2))) not in lnodep:
                    new_nodep = np.append(
                        new_nodep, np.array(line1.intersection(line2))[None], axis=0
                    )
                    intersects_idx = [list(new_edges[i]), list(new_edges[j])]
                    new_edges.pop(new_edges.index(intersects_idx[0]))
                    new_edges.pop(new_edges.index(intersects_idx[1]))

                    for n in np.array(intersects_idx).flatten():
                        new_edges.append([n, len(new_nodep) - 1])
                    break

        multiline = MultiLineString([LineString(new_nodep[e]) for e in new_edges])
        lnodep = new_nodep.tolist()
    return new_nodep, np.array(new_edges)


@nb.njit
def mahalanobis(M, cent):
    cov = np.cov(M, rowvar=0)
    try:
        cov_inverse = np.linalg.inv(cov)
    except:
        cov_inverse = np.linalg.pinv(cov)

    M_c = M - cent
    dist = np.sqrt(np.sum((M_c) * cov_inverse.dot(M_c.T).T, axis=1))
    return dist


@nb.njit
def polygon_area(x, y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def pp_compactness(cycle_nodep):
    """Polsby-Popper compactness"""
    area = polygon_area(cycle_nodep[:, 0], cycle_nodep[:, 1])
    length = np.sum(np.sqrt((np.diff(cycle_nodep, axis=0) ** 2).sum(axis=1)))
    return (4 * np.pi * area) / (length ** 2)


def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(
                reversed(cycle[mi_plus_1:])
            )
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


def in_hull(points, queries):
    hull = _Qhull(
        b"i",
        points,
        options=b"",
        furthest_site=False,
        incremental=False,
        interior_point=None,
    )
    equations = hull.get_simplex_facet_array()[2].T
    return np.all(queries @ equations[:-1] < -equations[-1], axis=1)


def add_loops(
    adata,
    min_path_len=None,
    nnodes=None,
    max_inner_fraction=0.05,
    min_node_n_points=5,
    max_n_points=np.inf,
    # max_empty_curve_fraction=.2,
    min_compactness=0.5,
    radius=None,
    allow_same_branch=True,
    fit_loops=True,
    Lambda=0.02,
    Mu=0.1,
    weights=None,
    plot=False,
    verbose=False,
    key="epg",
):

    # --- Init parameters, variables
    X = _get_graph_data(adata, key)
    init_nodes_pos = adata.uns[key]["node_pos"]
    init_edges = adata.uns[key]["edge"]
    epg = nx.convert_matrix.from_scipy_sparse_matrix(adata.uns[key]["conn"])

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, init_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )
    leaves = [k for k, v in epg.degree if v == 1]

    if radius is None:
        radius = np.mean(adata.uns[key["edge_len"]]) * min_path_len
        # scipy.spatial.distance.pdist(init_nodes_pos[leaves]))
    if min_path_len is None:
        min_path_len = len(init_nodes_pos) // 6
    if weights is None:
        weights = np.ones(len(X))[:, None]
    if nnodes is None:
        nnodes = min(20, max(8, len(init_nodes_pos) // 6))

    # --- Get candidate nodes to connect
    dist, ind = (
        NearestNeighbors(radius=radius)
        .fit(init_nodes_pos)
        .radius_neighbors(init_nodes_pos[leaves])
    )
    net = elpigraph.src.graphs.ConstructGraph({"Edges": [init_edges]})

    if all(np.array(net.degree()) <= 2):
        branches = net.get_shortest_paths(leaves[0], leaves[-1])
    else:
        (
            dict_tree,
            dict_branches,
            dict_branches_single_end,
        ) = elpigraph.src.supervised.get_tree(init_edges, leaves[0])
        branches = list(dict_branches.values())

    candidate_nodes = []
    for i in range(len(leaves)):
        root_branch = [b for b in branches if leaves[i] in b][0]

        if allow_same_branch:
            _cand_nodes = [node for b in branches for node in b if node in ind[i]]
        else:
            _cand_nodes = [
                node
                for b in branches
                for node in b
                if not (node in root_branch) and node in ind[i]
            ]
        paths = net.get_shortest_paths(leaves[i], _cand_nodes)
        candidate_nodes.append([p[-1] for p in paths if len(p) > min_path_len])

    # --- Test each of the loops connecting a leaf to its candidate nodes,
    # --- for each leaf select the one with minimum energy and that respect constraints
    if verbose:
        print("testing", sum([len(_) for _ in candidate_nodes]), "candidates")

    new_edges = []
    new_nodep = []
    new_leaves = []
    new_part = []
    new_energy = []
    for i, l in enumerate(leaves):
        energies = []
        merged_edges = []
        merged_nodep = []
        merged_part = []
        loop_edges = []
        loop_nodep = []
        loop_leaves = []
        for c in candidate_nodes[i]:

            clus = (part == c) | (part == l)
            X_fit = np.vstack((init_nodes_pos[c], init_nodes_pos[l], X[clus.flat]))
            try:
                _adata = sc.AnnData(X_fit)
                learn_graph(
                    _adata,
                    method="principal_curve",
                    obsm=None,
                    use_seed=False,
                    epg_lambda=Lambda,
                    epg_mu=Mu,
                    n_nodes=nnodes,
                    FixNodesAtPoints=[[0], [1]],
                )
            except Exception as e:
                energies.append(np.inf)
                merged_edges.append(np.inf)
                merged_nodep.append(np.inf)
                loop_edges.append(np.inf)
                loop_nodep.append(np.inf)
                loop_leaves.append(np.inf)
                # candidate curve has infinite energy, ignore error
                if e.args == (
                    "local variable 'NewNodePositions' referenced before assignment",
                ):
                    continue
                else:
                    raise e

            # ---get nodep, edges, create new graph with added loop
            nodep, edges = _adata.uns["epg"]["node_pos"], _adata.uns["epg"]["edge"]
            # _part, _part_dist = elpigraph.src.core.PartitionData(
            #    X_fit, nodep, 10 ** 6, np.sum(X_fit ** 2, axis=1, keepdims=1)
            # )
            _edges = edges.copy()
            _edges[(edges != 0) & (edges != 1)] += init_edges.max() - 1
            _edges[edges == 0] = c
            _edges[edges == 1] = l
            _merged_edges = np.concatenate((init_edges, _edges))
            _merged_nodep = np.concatenate((init_nodes_pos, nodep[2:]))

            cycle_edges = find_all_cycles(nx.Graph(_merged_edges.tolist()))[0]

            Mus = np.repeat(Mu, len(_merged_nodep))
            Mus[cycle_edges] = Mu / 10000
            ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(
                _merged_edges, Lambdas=Lambda, Mus=Mus
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
            _adata.uns["epg"]["node_pos"][2:] = nodep[2:] = _merged_nodep[
                len(init_nodes_pos) :
            ]

            ### candidate validity tests ###
            valid = True
            # --- curve validity test

            # if (max_empty_curve_fraction is not None) and valid: # if X_fit projected to curve has long gaps
            #    infer_pseudotime(_adata,source=0)
            #    sorted_X_proj=_adata.obsm['X_epg_proj'][_adata.obs['epg_pseudotime'].argsort()]
            #    dist = np.sqrt((np.diff(sorted_X_proj,axis=0)**2).sum(axis=1))
            #    curve_len = np.sum(_adata.uns['epg']['edge_len'])
            #    if np.max(dist) > (curve_len * max_empty_curve_fraction):
            #        valid = False

            # --- cycle validity test
            if valid:
                G = nx.Graph(_merged_edges.tolist())
                cycle_edges = find_all_cycles(G)[0]
                cycle_nodep = np.array([_merged_nodep[e] for e in cycle_edges])
                cent_part, cent_dists = elpigraph.src.core.PartitionData(
                    X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                )
                cycle_points = np.isin(cent_part.flat, cycle_edges)

                if X.shape[1] > 2:
                    pca = PCA(n_components=2, svd_solver="arpack").fit(X[cycle_points])
                    X_cycle_2d = pca.transform(X[cycle_points])
                    cycle_2d = pca.transform(cycle_nodep)
                else:
                    cycle_2d = cycle_nodep
                    X_cycle_2d = X[cycle_points]

                inside_idx = in_hull(cycle_2d, X_cycle_2d)

                if sum(inside_idx) == 0:
                    inner_fraction == 0.0
                else:
                    cycle_centroid = np.mean(cycle_2d, axis=0, keepdims=1)
                    X_inside = X_cycle_2d[inside_idx]

                    w = mahalanobis(X_inside, cycle_centroid)

                    # points belonging to cycle shrunk by 10% or within 2 std of centroid (mahalanobis < 2)
                    shrunk_cycle_2d = shrink_or_swell_shapely_polygon(
                        cycle_2d, factor=0.1
                    )

                    # prevent shapely bugs when multi-polygon is returned. Fall back to mahalanobis
                    if type(shrunk_cycle_2d) == MultiPolygon:
                        in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                    else:
                        shrunk_cycle_2d = np.array(shrunk_cycle_2d.exterior.coords)

                        # prevent bug when self-intersection
                        if len(shrunk_cycle_2d) == 0:
                            in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                        else:
                            in_shrunk_cycle = in_hull(shrunk_cycle_2d, X_inside)
                    idx_close = in_shrunk_cycle | (w < 1)
                    w = 1 - w / w.max()
                    w[idx_close] = 1

                    # cycle_edges_array = np.append(np.array(list(zip(range(len(cycle_2d)-1),
                    #                                  range(1,len(cycle_2d))))),[[len(cycle_2d)-1,0]],axis=0)
                    # w, idx_close = get_weights_lineproj(X_inside,cycle_2d,cycle_edges_array,cycle_centroid[0],threshold=.2)

                    inner_fraction = np.sum(w) / np.sum(cycle_points)

                if init_nodes_pos.shape[1] == 2:
                    intersect = not (
                        MultiLineString(
                            [LineString(_merged_nodep[e]) for e in _merged_edges]
                        ).is_simple
                    )
                    if intersect:
                        valid = False

                if (
                    any(
                        np.bincount(cent_part.flat, minlength=len(_merged_nodep))[
                            len(init_nodes_pos) :
                        ]
                        < min_node_n_points
                    )  # if empty cycle node
                    or (
                        inner_fraction > max_inner_fraction
                    )  # if high fraction of points inside
                    or (not np.isfinite(inner_fraction))  # prevent no points error
                    or (np.sum(idx_close) > max_n_points)  # if too many points inside
                    or pp_compactness(cycle_2d) < min_compactness
                ):  # if loop is very narrow
                    valid = False

            # _merged_part,_merged_part_dist=elpigraph.src.core.PartitionData(X,_merged_nodep,10**6,SquaredX=SquaredX)
            # X_proj = elpigraph.src.reporting.project_point_onto_graph(X, _merged_nodep, _merged_edges, _merged_part)['X_projected']
            # dist2proj = np.linalg.norm(X - X_proj, axis=1)
            # ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(_merged_edges, Lambdas=Lambda, Mus=Mu)
            # ElasticEnergy, MSE, EP, RP = elpigraph.src.core.ComputePenalizedPrimitiveGraphElasticEnergy(_merged_nodep,ElasticMatrix,dist2proj,alpha=0.01,beta=0)
            ###
            # plt.title(f'{c}, {l}, VALID={valid}, MSE={MSE:.4f}, EP={EP:.4f}, RP={RP:.4f}, \n inner%={inner_fraction:.2f},'+str(round(np.sum(w),2))+' '+str(np.sum(cycle_points)))
            # plt.scatter(*X[:,:2].T,alpha=.1,s=5)
            ##plt.scatter(*X_fit[:,:2].T,s=5)
            ##plt.scatter(*cycle_centroid[:,:2].T,s=50,c='red')
            # try:
            #    plt.scatter(*X[cycle_points][inside_idx,:2].T,alpha=1,s=5,c=w.flat);plt.colorbar()
            # except:
            #    plt.scatter(*X[cycle_points][inside_idx,:2].T,alpha=1,s=5)
            #
            # for e in _merged_edges:
            #    plt.plot([_merged_nodep[e[0],0],_merged_nodep[e[1],0]],[_merged_nodep[e[0],1],_merged_nodep[e[1],1]],c='k')
            # plt.show()
            # inner_fraction = np.sum(w)/np.sum(cycle_points)

            # ---> if cycle is invalid, continue
            if not valid:
                energies.append(np.inf)
                merged_edges.append(np.inf)
                merged_nodep.append(np.inf)
                merged_part.append(np.inf)
                loop_edges.append(np.inf)
                loop_nodep.append(np.inf)
                loop_leaves.append(np.inf)
                continue

            # ---> valid cycle, compute graph energy
            else:
                _merged_part, _merged_part_dist = elpigraph.src.core.PartitionData(
                    X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                )
                proj = elpigraph.src.reporting.project_point_onto_graph(
                    X, _merged_nodep, _merged_edges, _merged_part
                )
                MSE = proj["MSEP"]
                # dist2proj = np.sum(np.square(X - X_proj), axis=1)
                # ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(_merged_edges, Lambdas=Lambda, Mus=Mu)
                # ElasticEnergy, MSE, EP, RP = elpigraph.src.core.ComputePenalizedPrimitiveGraphElasticEnergy(_merged_nodep,
                #                                                                                            ElasticMatrix,
                #                                                                                            dist2proj,alpha=0.01,beta=0)

                energies.append(MSE)
                merged_edges.append(_merged_edges)
                merged_nodep.append(_merged_nodep)
                merged_part.append(np.where(np.isin(_merged_part.flat, cycle_edges))[0])
                loop_edges.append(edges)
                loop_nodep.append(nodep[2:])
                loop_leaves.append([c, l])

        # --- among all valid cycles found, retain the best
        if energies != [] and np.isfinite(energies).any():
            best = np.argmin(energies)
            if [loop_leaves[best][-1], loop_leaves[best][0]] not in new_leaves:
                # and not any(np.isin(loop_leaves[best],np.unique(np.array(new_leaves))))):

                new_edges.append(loop_edges[best])
                new_nodep.append(loop_nodep[best])
                new_leaves.append(loop_leaves[best])
                new_part.append(merged_part[best])
                new_energy.append(energies[best])
                _merged_edges = merged_edges[best]
                _merged_nodep = merged_nodep[best]

                if plot:
                    c = candidate_nodes[i][best]
                    clus = (part == c) | (part == l)
                    X_fit = np.vstack(
                        (init_nodes_pos[c], init_nodes_pos[l], X[clus.flat])
                    )
                    proj = elpigraph.src.reporting.project_point_onto_graph(
                        X, _merged_nodep, _merged_edges, _merged_part
                    )
                    MSE = proj["MSEP"]

                    # ----- cycle test
                    G = nx.Graph(_merged_edges.tolist())
                    cycle_edges = find_all_cycles(G)[0]
                    cycle_nodep = np.array([_merged_nodep[e] for e in cycle_edges])
                    cent_part, cent_dists = elpigraph.src.core.PartitionData(
                        X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                    )
                    cycle_points = np.isin(cent_part.flat, cycle_edges)

                    if X.shape[1] > 2:
                        pca = PCA(n_components=2, svd_solver="arpack").fit(cycle_nodep)
                        cycle_2d = pca.transform(cycle_nodep)
                        X_cycle_2d = pca.transform(X[cycle_points])
                    else:
                        cycle_2d = cycle_nodep
                        X_cycle_2d = X[cycle_points]
                    inside_idx = in_hull(cycle_2d, X_cycle_2d)

                    cycle_centroid = np.mean(cycle_2d, axis=0, keepdims=1)
                    X_inside = X_cycle_2d[inside_idx]

                    w = mahalanobis(X_inside, cycle_centroid)

                    # points belonging to cycle shrunk by 10% or within 2 std of centroid (mahalanobis < 2)
                    shrunk_cycle_2d = shrink_or_swell_shapely_polygon(
                        cycle_2d, factor=0.1
                    )

                    # prevent shapely bugs when multi-polygon is returned. Fall back to mahalanobis
                    if type(shrunk_cycle_2d) == MultiPolygon:
                        in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                    else:
                        shrunk_cycle_2d = np.array(shrunk_cycle_2d.exterior.coords)

                        # prevent bug when self-intersection
                        if len(shrunk_cycle_2d) == 0:
                            in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                        else:
                            in_shrunk_cycle = in_hull(shrunk_cycle_2d, X_inside)
                    idx_close = in_shrunk_cycle | (w < 1)
                    w = 1 - w / w.max()
                    w[idx_close] = 1

                    compactness = pp_compactness(cycle_2d)

                    plt.title(
                        f"{c}, {l}, MSE={MSE:.4f}, \n inner%={inner_fraction:.2f}, compactness={compactness:.2f}"
                    )
                    plt.scatter(*X[:, :2].T, alpha=0.1, s=5)
                    plt.scatter(*X_fit[:, :2].T, s=5)
                    for e in _merged_edges:
                        plt.plot(
                            [_merged_nodep[e[0], 0], _merged_nodep[e[1], 0]],
                            [_merged_nodep[e[0], 1], _merged_nodep[e[1], 1]],
                            c="k",
                        )

                    _ = plt.scatter(*X[cycle_points][inside_idx, :2].T, c=w.flat, s=5)
                    plt.colorbar(_)

                    plt.show()

    # ignore equivalent loops (with more than 2/3 shared points)
    valid = np.ones(len(new_part))
    for i in range(len(new_part) - 1):
        for j in range(i + 1, len(new_part)):
            if (
                len(np.intersect1d(new_part[i], new_part[j]))
                / min(len(new_part[i]), len(new_part[j]))
            ) > (2 / 3):
                if np.argmin([new_energy[i], new_energy[j]]) == 0:
                    valid[i] = 0
                else:
                    valid[j] = 0

    new_edges = [e for i, e in enumerate(new_edges) if valid[i]]
    new_nodep = [e for i, e in enumerate(new_nodep) if valid[i]]
    new_leaves = [e for i, e in enumerate(new_leaves) if valid[i]]
    new_part = [e for i, e in enumerate(new_part) if valid[i]]
    new_energy = [e for i, e in enumerate(new_energy) if valid[i]]

    ### form graph with all valid loops found ###
    if (new_edges == []) or (sum(valid) == 0):
        return (None, None, None, None, None)

    for i, loop_edges in enumerate(new_edges):
        if i == 0:
            loop_edges[(loop_edges != 0) & (loop_edges != 1)] += init_edges.max() - 1
            loop_edges[loop_edges == 0] = new_leaves[i][0]
            loop_edges[loop_edges == 1] = new_leaves[i][1]
            merged_edges = np.concatenate((init_edges, loop_edges))
        else:
            loop_edges[(loop_edges != 0) & (loop_edges != 1)] += merged_edges.max() - 1
            loop_edges[loop_edges == 0] = new_leaves[i][0]
            loop_edges[loop_edges == 1] = new_leaves[i][1]
            merged_edges = np.concatenate((merged_edges, loop_edges))
    merged_nodep = np.concatenate((init_nodes_pos, *new_nodep))

    ### optionally refit the entire graph ###
    if fit_loops:
        cycle_edges = np.concatenate(find_all_cycles(nx.Graph(merged_edges.tolist())))
        Mus = np.repeat(Mu, len(merged_nodep))
        Mus[cycle_edges] = Mu / 10000
        ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(
            merged_edges, Lambdas=Lambda, Mus=Mus
        )
        (
            merged_nodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X, merged_nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[]
        )
        # check intersection
        if merged_nodep.shape[1] == 2:
            intersect = not (
                MultiLineString(
                    [LineString(merged_nodep[e]) for e in merged_edges]
                ).is_simple
            )
            if intersect:
                merged_nodep, merged_edges = remove_intersections(
                    merged_nodep, merged_edges
                )

    return new_edges, new_nodep, new_leaves, merged_nodep, merged_edges
