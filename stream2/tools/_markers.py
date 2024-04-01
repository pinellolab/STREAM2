import numpy as np
import numba as nb
import pandas as pd
import math
import scipy
import os
from copy import deepcopy
from statsmodels.sandbox.stats.multicomp import multipletests

from .. import _utils
from .._settings import settings


def spearman_columns(A, B):
    """Spearman correlation over columns
    A,B: np.arrays with same shape

    Returns
    -------
    correlations: np.array
        correlations[i] = spearman_corrcoef(A[:,i],B[:,i])
    """
    assert A.shape == B.shape
    return pearson_corr(rankdata(A.T), rankdata(B.T))


def spearman_pairwise(A, B):
    """Spearman correlation matrix
    A,B: np.arrays with same shape

    Returns
    -------
    correlations: np.array
        correlations[i,j] = spearman_corrcoef(A[:,i],B[:,j])
    """
    n, m = A.shape[1], B.shape[1]
    i, j = np.ones((n, m)).nonzero()
    return pearson_corr(rankdata(A.T[i]), rankdata(B.T[j])).reshape(n, m)


@nb.njit(parallel=True)
def xicorr_columns(A, B):
    """XI correlation over columns
    A,B: 2d np.arrays with same shape

    Returns
    -------
    correlations:
        correlations[i] = xi_corrcoef(A[:,i],B[:,i])
    """
    assert A.shape == B.shape
    n, m = A.shape
    corrs = np.zeros(m)
    pvals = np.zeros(m)
    for i in nb.prange(m):
        corrs[i], pvals[i] = xicorr(A[:, i], B[:, i], n)
    return corrs, pvals


@nb.njit(parallel=True)
def xicorr_pairwise(A, B):
    """XI correlation over columns
    A,B: 2d np.arrays with same shape

    Returns
    -------
    correlations:
        correlations[i] = xi_corrcoef(A[:,i],B[:,i])
    """
    assert A.shape == B.shape
    ns = len(A)

    n, m = A.shape[1], B.shape[1]

    corrs = np.ones((n, m))
    pvals = np.ones((n, m))
    for i in nb.prange(n):
        for j in nb.prange(m):
            corrs[i, j], pvals[i, j] = xicorr(A[:, i], B[:, j], ns)
    return corrs, pvals


@nb.njit
def _nb_unique1d(ar):
    """Numba speedup."""
    ar = ar.flatten()
    perm = ar.argsort(kind="mergesort")
    aux = ar[perm]

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if (
            aux.dtype.kind == "c"
        ):  # for complex all NaNs are considered equivalent
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
def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@nb.njit
def average_ties(X):
    """Same as scipy.stats.rankdata method="average"."""
    xi = np.argsort(X)
    xi_rank = np.argsort(xi)
    unique, _, inverse, c_ = _nb_unique1d(X)
    unique_rank_sum = np.zeros_like(unique)
    for i0, inv in enumerate(inverse):
        unique_rank_sum[inv] += xi_rank[i0]
    unique_count = np.zeros_like(unique)
    for i0, inv in enumerate(inverse):
        unique_count[inv] += 1
    unique_rank_mean = unique_rank_sum / unique_count
    rank_mean = unique_rank_mean[inverse]
    return rank_mean + 1


def average_stat(
    mdata,
    transition_markers,
):
    """Average correlation coefficients
    doi.org/10.3758/BF03334037
    """
    counts = pd.Series(
        {
            assay: sum(~np.isnan(adata.obs["epg_pseudotime"]))
            for assay, adata in mdata.mod.items()
        }
    )

    joint_index = set.intersection(
        *[set(df.index) for df in transition_markers.values()]
    )
    joint_dfs = {
        assay: df.loc[joint_index] for assay, df in transition_markers.items()
    }

    joint_stats = pd.concat(
        [df["stat"] for assay, df in joint_dfs.items()], axis=1
    )
    joint_stats.columns = joint_dfs.keys()

    avg_stat = np.tanh(
        np.sum(
            np.arctanh(joint_stats)
            * (counts - 3)
            / (sum(counts) - 3 * len(counts)),
            axis=1,
        )
    )
    return avg_stat


@nb.njit
def xicorr(x, y, n):
    """Translated from R https://github.com/cran/XICOR/"""
    # ---corr
    PI = average_ties(x)
    fr = average_ties(y) / n
    gr = average_ties(-y) / n
    fr = fr[np.argsort(PI, kind="mergesort")]

    CU = np.mean(gr * (1 - gr))
    A1 = np.abs(np.diff(fr)).sum() / (2 * n)
    xi = 1 - A1 / CU

    # ---pval
    qfr = np.sort(fr)
    ind = np.arange(n) + 1
    ind2 = np.array([2 * n - 2 * ind[i - 1] + 1 for i in ind])

    ai = np.mean(ind2 * qfr * qfr) / n
    ci = np.mean(ind2 * qfr) / n
    cq = np.cumsum(qfr)

    m = (cq + (n - ind) * qfr) / n
    b = np.mean(m ** 2)
    v = (ai - 2 * b + np.square(ci)) / np.square(CU)

    # sd = np.sqrt(v/n)
    pval = 1 - normal_cdf(np.sqrt(n) * xi / np.sqrt(v))
    return xi, pval


@nb.njit(parallel=True)
def xicorr_ps(x, Y):
    """Fast xi correlation coefficient
    x: 0d np.array
    Y: 2d np.array
    """
    n = len(Y)
    corrs = np.zeros(Y.shape[1])
    pvals = np.zeros(Y.shape[1])
    for i in nb.prange(Y.shape[1]):
        corrs[i], pvals[i] = xicorr(x,Y[:, i], n)
    return corrs, pvals


def spearman_ps(x, Y):
    """Fast spearman correlation coefficient
    X: 2d np.array
    y: 0d np.array
    """
    return pearson_corr(rankdata(x[None]), rankdata(Y.T))


def pearson_corr(arr1, arr2):
    """Pearson correlation along the last dimension of two multidimensional
    arrays."""
    mean1 = np.mean(arr1, axis=-1, keepdims=1)
    mean2 = np.mean(arr2, axis=-1, keepdims=1)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)

    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    )


@nb.njit(parallel=True, fastmath=True)
def rankdata(X):
    """reimplementing scipy.stats.rankdata faster."""
    tmp = np.zeros_like(X)
    for i in nb.prange(X.shape[0]):
        tmp[i] = _rankdata_inner(X[i])
    return tmp


@nb.njit
def _rankdata_inner(x):
    """inner loop for rankdata."""
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


def scale_marker_expr(df_marker_detection, percentile_expr):
    # optimal version for STREAM1
    ind_neg = df_marker_detection.min() < 0
    ind_pos = df_marker_detection.min() >= 0
    df_neg = df_marker_detection.loc[:, ind_neg]
    df_pos = df_marker_detection.loc[:, ind_pos]

    if ind_neg.sum() > 0:
        print("Matrix contains negative values...")
        # genes with negative values
        minValues = df_neg.apply(
            lambda x: np.percentile(x[x < 0], 100 - percentile_expr), axis=0
        )
        maxValues = df_neg.apply(
            lambda x: np.percentile(x[x > 0], percentile_expr), axis=0
        )
        for i in range(df_neg.shape[1]):
            df_gene = df_neg.iloc[:, i].copy(deep=True)
            df_gene[df_gene < minValues[i]] = minValues[i]
            df_gene[df_gene > maxValues[i]] = maxValues[i]
            df_neg.iloc[:, i] = df_gene - minValues[i]
        df_neg = df_neg.copy(deep=True)
        maxValues = df_neg.max(axis=0)
        df_neg_scaled = df_neg / np.array(maxValues)[:, None].T
    else:
        df_neg_scaled = pd.DataFrame(index=df_neg.index)

    if ind_pos.sum() > 0:
        maxValues = df_pos.apply(
            lambda x: np.percentile(x[x > 0], percentile_expr), axis=0
        )
        df_pos_scaled = df_pos / np.array(maxValues)[:, None].T
        df_pos_scaled[df_pos_scaled > 1] = 1
    else:
        df_pos_scaled = pd.DataFrame(index=df_pos.index)

    df_marker_detection_scaled = pd.concat(
        [df_neg_scaled, df_pos_scaled], axis=1
    )

    return df_marker_detection_scaled


def detect_transition_markers(
    adata,
    percentile_expr=95,
    min_num_cells=5,
    fc_cutoff=1,
    method="spearman",
    key="epg",
):

    file_path = os.path.join(settings.workdir, "transition_markers")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Extract cells by parameters in previous infer_pseudotime() step
    path_source = adata.uns[f"{key}_pseudotime_params"]["source"]
    path_target = adata.uns[f"{key}_pseudotime_params"]["target"]
    nodes_to_include_path = adata.uns[f"{key}_pseudotime_params"][
        "nodes_to_include"
    ]

    if path_target is None:
        print(
            "Please re-run infer_pseudotime() and specify value for "
            "parameter target"
        )
        exit()

    cells, path_alias = _utils.get_path(
        adata, path_source, path_target, nodes_to_include_path, key
    )

    # Scale matrix with expressed markers
    input_markers = adata.var_names.tolist()
    if scipy.sparse.issparse(adata.X):
        mat = adata[:, input_markers].X.todense()
    else:
        mat = adata[:, input_markers].X

    df_sc = pd.DataFrame(
        index=adata.obs_names.tolist(),
        data=mat,
        columns=input_markers,
    )

    print(
        "Filtering out markers that are expressed in less than "
        + str(min_num_cells)
        + " cells ..."
    )
    input_markers_expressed = np.array(input_markers)[
        np.where((df_sc[input_markers]>0).sum(axis=0) > min_num_cells)[0]
    ].tolist()
    df_marker_detection = df_sc[input_markers_expressed].copy()

    df_scaled_marker_expr = scale_marker_expr(
        df_marker_detection, percentile_expr
    )
    adata.uns["scaled_marker_expr"] = df_scaled_marker_expr

    print(str(len(input_markers_expressed)) + " markers are being scanned ...")

    df_cells = deepcopy(df_scaled_marker_expr.loc[cells])
    pseudotime_cells = adata.obs[f"{key}_pseudotime"][cells]
    df_cells_sort = df_cells.iloc[np.argsort(pseudotime_cells)]
    pseudotime_cells_sort = pseudotime_cells[np.argsort(pseudotime_cells)]

    dict_tg_edges = dict()

    id_initial = range(0, int(df_cells_sort.shape[0] * 0.2))
    id_final = range(
        int(df_cells_sort.shape[0] * 0.8), int(df_cells_sort.shape[0] * 1)
    )
    values_initial, values_final = (
        df_cells_sort.iloc[id_initial, :],
        df_cells_sort.iloc[id_final, :],
    )
    diff_initial_final = np.abs(
        values_final.mean(axis=0) - values_initial.mean(axis=0)
    )

    # original expression
    df_cells_ori = deepcopy(df_marker_detection.loc[cells])
    df_cells_sort_ori = df_cells_ori.iloc[np.argsort(pseudotime_cells)]
    values_initial_ori, values_final_ori = (
        df_cells_sort_ori.iloc[id_initial, :],
        df_cells_sort_ori.iloc[id_final, :],
    )

    ix_pos = diff_initial_final > 0
    logfc = pd.Series(
        np.zeros(len(diff_initial_final)), index=diff_initial_final.index
    )
    logfc[ix_pos] = np.log2(
        (
            np.maximum(values_final.mean(axis=0), values_initial.mean(axis=0))
            + 0.01
        )
        / (
            np.minimum(values_final.mean(axis=0), values_initial.mean(axis=0))
            + 0.01
        )
    )

    ix_cutoff = np.array(logfc > fc_cutoff)

    if sum(ix_cutoff) == 0:
        print(
            "No Transition markers are detected in branch with nodes "
            + str(path_source)
            + " to "
            + str(path_target)
        )

    else:
        df_stat_pval_qval = pd.DataFrame(
            np.full((sum(ix_cutoff), 8), np.nan),
            columns=[
                "stat",
                "logfc",
                "pval",
                "qval",
                "initial_mean",
                "final_mean",
                "initial_mean_ori",
                "final_mean_ori",
            ],
            index=df_cells_sort.columns[ix_cutoff],
        )

        if method == "spearman":
            df_stat_pval_qval["stat"] = spearman_ps(
                np.array(pseudotime_cells_sort),
                np.array(df_cells_sort.iloc[:, ix_cutoff]),
            )
            df_stat_pval_qval["pval"] = p_val(
                df_stat_pval_qval["stat"], len(pseudotime_cells_sort)
            )
        elif method == "xi":
            # /!\ dont use df_cells_sort
            # and pseudotime_cells_sort, breaks xicorr
            res = xicorr_ps(
                np.array(pseudotime_cells),
                np.array(df_cells.iloc[:, ix_cutoff])
            )
            df_stat_pval_qval["stat"] = res[0]
            df_stat_pval_qval["pval"] = res[1]
        else:
            raise ValueError("method must be one of 'spearman', 'xi'")

        df_stat_pval_qval["logfc"] = logfc
        p_values = df_stat_pval_qval["pval"]
        q_values = multipletests(p_values, method="fdr_bh")[1]
        df_stat_pval_qval["qval"] = q_values
        df_stat_pval_qval["initial_mean"] = values_initial.mean(axis=0)
        df_stat_pval_qval["final_mean"] = values_final.mean(axis=0)
        df_stat_pval_qval["initial_mean_ori"] = values_initial_ori.mean(axis=0)
        df_stat_pval_qval["final_mean_ori"] = values_final_ori.mean(axis=0)

        dict_tg_edges[path_alias] = df_stat_pval_qval.sort_values(["qval"])

        dict_tg_edges[path_alias].to_csv(
            os.path.join(
                file_path,
                "transition_markers_path_"
                + str(path_source)
                + "-"
                + str(path_target)
                + ".tsv",
            ),
            sep="\t",
            index=True,
        )

    if "transition_markers" in adata.uns.keys():
        adata.uns["transition_markers"].update(dict_tg_edges)
    else:
        adata.uns["transition_markers"] = dict_tg_edges
