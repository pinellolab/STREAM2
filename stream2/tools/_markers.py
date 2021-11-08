import numpy as np
import numba as nb
import pandas as pd
import scipy
import multiprocessing
import os
from copy import deepcopy
from statsmodels.sandbox.stats.multicomp import multipletests


@nb.njit
def nb_unique1d(ar):
    """
    Numba speedup
    """
    ar = ar.flatten()
    perm = ar.argsort(kind="mergesort")
    aux = ar[perm]

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side="left")
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side="left")
        mask[1:aux_firstnan] = aux[1:aux_firstnan] != aux[: aux_firstnan - 1]
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1 :] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask
    idx = np.append(np.nonzero(mask)[0], mask.size)

    # idx      #inverse   #counts
    return aux[mask], perm[mask], inv_idx, np.diff(idx)


@nb.njit
def _xicorr(X, Y):
    """xi correlation coefficient
    X,Y 0 dimensional np.arrays"""
    n = X.size
    xi = np.argsort(X, kind="quicksort")
    Y = Y[xi]
    _, _, b, c = nb_unique1d(Y)
    r = np.cumsum(c)[b]
    _, _, b, c = nb_unique1d(-Y)
    l = np.cumsum(c)[b]
    return 1 - n * np.abs(np.diff(r)).sum() / (2 * (l * (n - l)).sum())


@nb.njit
def _xicorr_inner(X, Y, n):
    """Numba fast xi correlation coefficient
    X,Y 0 dimensional np.arrays"""
    xi = np.argsort(X, kind="quicksort")
    Y = Y[xi]
    _, _, b, c = nb_unique1d(Y)
    r = np.cumsum(c)[b]
    _, _, b, c = nb_unique1d(-Y)
    l = np.cumsum(c)[b]
    return 1 - n * np.abs(np.diff(r)).sum() / (2 * (l * (n - l)).sum())


@nb.njit(parallel=True)
def _xicorr_loop_parallel(X, Y):
    """Numba fast parallel xi correlation coefficient
    X,Y 0 dimensional np.arrays"""
    n = len(X)
    corrs = np.zeros(X.shape[1])
    for i in nb.prange(X.shape[1]):
        corrs[i] = _xicorr_inner(X[:, i], Y, n)
    return corrs


def nb_spearman(x, Y):
    """
    Fast equivalent to 
    for i in range(y.shape[1]): spearmanr(x,y[:,i]).correlation
    """
    return pearson_corr(_rankdata(x[None]), _rankdata(Y.T))


def pearson_corr(arr1, arr2):
    """
    Pearson correlation along the last dimension of two multidimensional arrays.
    """
    mean1 = np.mean(arr1, axis=-1, keepdims=1)
    mean2 = np.mean(arr2, axis=-1, keepdims=1)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)

    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0))


@nb.njit(parallel=True, fastmath=True)
def _rankdata(X):
    """reimplementing scipy.stats.rankdata faster """
    tmp = np.zeros_like(X)
    for i in nb.prange(X.shape[0]):
        tmp[i] = _rankdata_inner(X[i])
    return tmp


@nb.njit
def _rankdata_inner(x):
    """inner loop for _rankdata"""
    sorter = np.argsort(x)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    x = x[sorter]
    obs = np.concatenate((np.array([True]), x[1:] != x[:-1]))
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.append(np.nonzero(obs)[0], len(obs))
    # average method
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def p_val(r, n):
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return scipy.stats.t.sf(np.abs(t), n - 1) * 2


def scale_marker_expr(params):
    df_marker_detection = params[0]
    marker = params[1]
    percentile_expr = params[2]
    marker_values = df_marker_detection[marker].copy()
    if min(marker_values) < 0:
        min_marker_values = np.percentile(
            marker_values[marker_values < 0], 100 - percentile_expr
        )
        marker_values[marker_values < min_marker_values] = min_marker_values
        max_marker_values = np.percentile(
            marker_values[marker_values > 0], percentile_expr
        )
        marker_values[marker_values > max_marker_values] = max_marker_values
        marker_values = marker_values - min(marker_values)
    else:
        max_marker_values = np.percentile(
            marker_values[marker_values > 0], percentile_expr
        )
        marker_values[marker_values > max_marker_values] = max_marker_values
    marker_values = marker_values / max_marker_values
    return marker_values


def detect_transition_markers(
    adata,
    marker_list=None,
    cutoff_spearman=0.4,
    cutoff_logfc=0.25,
    percentile_expr=95,
    n_jobs=1,
    min_num_cells=5,
    use_precomputed=True,
    key="epg",
):

    file_path = os.path.join(adata.uns["workdir"], "transition_markers")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    source = adata.uns[f"{key}_pseudotime_params"]["source"]
    target = adata.uns[f"{key}_pseudotime_params"]["target"]
    nodes_to_include = adata.uns[f"{key}_pseudotime_params"]["nodes_to_include"]

    cells = adata.obs_names[~np.isnan(adata.obs[f"{key}_pseudotime"])]
    path_alias = "Path_%s-%s-%s" % (source, nodes_to_include, target)

    if marker_list is None:
        print("Scanning all features ...")
        marker_list = adata.var_names.tolist()
    else:
        print("Scanning the specified marker list ...")
        ###remove duplicate keys
        marker_list = list(dict.fromkeys(marker_list))
        for marker in marker_list:
            if marker not in adata.var_names:
                raise ValueError("could not find '%s' in `adata.var_names`" % (marker))

    #### Scale matrix with expressed markers
    if use_precomputed and ("scaled_marker_expr" in adata.uns.keys()):
        print("Importing precomputed scaled marker expression matrix ...")
        results = adata.uns["scaled_marker_expr"]
        df_results = pd.DataFrame(results).T
        if all(np.isin(marker_list, df_results.columns.tolist())):
            input_markers_expressed = marker_list
            df_scaled_marker_expr = df_results[input_markers_expressed]
        else:
            raise ValueError(
                "Not all markers in `marker_list` can be found in precomputed scaled marker expression matrix. Please set `use_precomputed=False`"
            )

    else:
        input_markers = marker_list
        df_sc = pd.DataFrame(
            index=adata.obs_names.tolist(),
            data=adata[:, input_markers].X,
            columns=input_markers,
        )
        print(
            "Filtering out markers that are expressed in less than "
            + str(min_num_cells)
            + " cells ..."
        )
        input_markers_expressed = np.array(input_markers)[
            np.where((df_sc[input_markers] > 0).sum(axis=0) > min_num_cells)[0]
        ].tolist()
        df_marker_detection = df_sc[input_markers_expressed].copy()
        print(str(n_jobs) + " cpus are being used ...")
        params = [
            (df_marker_detection, x, percentile_expr) for x in input_markers_expressed
        ]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_marker_expr, params)
        pool.close()
        adata.uns["scaled_marker_expr"] = results
        df_scaled_marker_expr = pd.DataFrame(results).T

    print(str(len(input_markers_expressed)) + " markers are being scanned ...")

    df_cells = deepcopy(df_scaled_marker_expr.loc[cells])
    pseudotime_cells = adata.obs[f"{key}_pseudotime"][cells]
    df_cells_sort = df_cells.iloc[np.argsort(pseudotime_cells)]

    dict_tg_edges = dict()

    id_initial = range(0, int(df_cells_sort.shape[0] * 0.2))
    id_final = range(int(df_cells_sort.shape[0] * 0.8), int(df_cells_sort.shape[0] * 1))
    values_initial, values_final = (
        df_cells_sort.iloc[id_initial, :],
        df_cells_sort.iloc[id_final, :],
    )
    diff_initial_final = np.abs(values_final.mean(axis=0) - values_initial.mean(axis=0))

    ix_pos = diff_initial_final > 0
    logfc = pd.Series(np.zeros(len(diff_initial_final)), index=diff_initial_final.index)
    logfc[ix_pos] = np.log2(
        np.maximum(values_final.mean(axis=0), values_initial.mean(axis=0))
        / (
            np.minimum(values_final.mean(axis=0), values_initial.mean(axis=0))
            + diff_initial_final[ix_pos] / 1000.0
        )
    )

    ix_cutoff = np.array(logfc > cutoff_logfc)

    if sum(ix_cutoff) == 0:
        print(
            "No Transition markers are detected in branch with nodes "
            + str(source)
            + " to "
            + str(target)
        )

    else:
        df_stat_pval_qval = pd.DataFrame(
            np.full((sum(ix_cutoff), 4), np.nan),
            columns=["stat", "logfc", "pval", "qval"],
            index=df_cells_sort.columns[ix_cutoff],
        )
        df_stat_pval_qval["stat"] = nb_spearman(
            np.array(pseudotime_cells), np.array(df_cells_sort.iloc[:, ix_cutoff])
        )
        df_stat_pval_qval["logfc"] = logfc
        df_stat_pval_qval["pval"] = p_val(
            df_stat_pval_qval["stat"], len(pseudotime_cells)
        )

        p_values = df_stat_pval_qval["pval"]
        q_values = multipletests(p_values, method="fdr_bh")[1]
        df_stat_pval_qval["qval"] = q_values
        dict_tg_edges[path_alias] = df_stat_pval_qval[
            (abs(df_stat_pval_qval.stat) >= cutoff_spearman)
        ].sort_values(["qval"])
        dict_tg_edges[path_alias].to_csv(
            os.path.join(
                file_path,
                "transition_markers_path_" + str(source) + "-" + str(target) + ".tsv",
            ),
            sep="\t",
            index=True,
        )

    if "transition_markers" in adata.uns.keys():
        adata.uns["transition_markers"].update(dict_tg_edges)
    else:
        adata.uns["transition_markers"] = dict_tg_edges

