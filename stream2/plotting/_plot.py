"""plotting functions"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.core.dtypes.common import is_numeric_dtype
import seaborn as sns
from adjustText import adjust_text
from pandas.api.types import (
    is_string_dtype,
    is_categorical_dtype,
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

from .._settings import settings
from ._utils import generate_palette
from .. import _utils


def violin(
    adata,
    list_obs=None,
    list_var=None,
    jitter=0.4,
    size=1,
    log=False,
    pad=1.08,
    w_pad=None,
    h_pad=3,
    fig_size=(3, 3),
    fig_ncol=3,
    save_fig=False,
    fig_path=None,
    fig_name="plot_violin.pdf",
    **kwargs,
):
    """Violin plot"""

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if obs not in adata.obs_keys():
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if var not in adata.var_keys():
            raise ValueError(f"could not find {var} in `adata.var_keys()`")
    if len(list_obs) > 0:
        df_plot = adata.obs[list_obs].copy()
        if log:
            df_plot = pd.DataFrame(
                data=np.log1p(df_plot.values),
                index=df_plot.index,
                columns=df_plot.columns,
            )
        fig_nrow = int(np.ceil(len(list_obs) / fig_ncol))
        fig = plt.figure(
            figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
        )
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
            sns.violinplot(ax=ax_i, y=obs, data=df_plot, inner=None, **kwargs)
            sns.stripplot(
                ax=ax_i, y=obs, data=df_plot, color="black", jitter=jitter, s=size
            )

            ax_i.set_title(obs)
            ax_i.set_ylabel("")
            ax_i.locator_params(axis="y", nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if save_fig:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(
                os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight"
            )
            plt.close(fig)
    if len(list_var) > 0:
        df_plot = adata.var[list_var].copy()
        if log:
            df_plot = pd.DataFrame(
                data=np.log1p(df_plot.values),
                index=df_plot.index,
                columns=df_plot.columns,
            )
        fig_nrow = int(np.ceil(len(list_obs) / fig_ncol))
        fig = plt.figure(
            figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
        )
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
            sns.violinplot(ax=ax_i, y=var, data=df_plot, inner=None, **kwargs)
            sns.stripplot(
                ax=ax_i, y=var, data=df_plot, color="black", jitter=jitter, s=size
            )

            ax_i.set_title(var)
            ax_i.set_ylabel("")
            ax_i.locator_params(axis="y", nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if save_fig:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(
                os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight"
            )
            plt.close(fig)


def hist(
    adata,
    list_obs=None,
    list_var=None,
    kde=True,
    size=1,
    log=False,
    pad=1.08,
    w_pad=None,
    h_pad=3,
    fig_size=(3, 3),
    fig_ncol=3,
    save_fig=False,
    fig_path=None,
    fig_name="plot_violin.pdf",
    **kwargs,
):
    """histogram plot"""

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if obs not in adata.obs_keys():
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if var not in adata.var_keys():
            raise ValueError(f"could not find {var} in `adata.var_keys()`")

    if len(list_obs) > 0:
        df_plot = adata.obs[list_obs].copy()
        if log:
            df_plot = pd.DataFrame(
                data=np.log1p(df_plot.values),
                index=df_plot.index,
                columns=df_plot.columns,
            )
        fig_nrow = int(np.ceil(len(list_obs) / fig_ncol))
        fig = plt.figure(
            figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
        )
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
            sns.histplot(ax=ax_i, x=obs, data=df_plot, kde=kde, **kwargs)
            ax_i.locator_params(axis="y", nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if save_fig:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(
                os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight"
            )
            plt.close(fig)
    if len(list_var) > 0:
        df_plot = adata.var[list_var].copy()
        if log:
            df_plot = pd.DataFrame(
                data=np.log1p(df_plot.values),
                index=df_plot.index,
                columns=df_plot.columns,
            )
        fig_nrow = int(np.ceil(len(list_obs) / fig_ncol))
        fig = plt.figure(
            figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
        )
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
            sns.histplot(ax=ax_i, x=var, data=df_plot, kde=kde, **kwargs)
            ax_i.locator_params(axis="y", nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if save_fig:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(
                os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight"
            )
            plt.close(fig)


def pca_variance_ratio(
    adata,
    log=True,
    show_cutoff=True,
    fig_size=(4, 4),
    save_fig=None,
    fig_path=None,
    fig_name="qc.pdf",
    pad=1.08,
    w_pad=None,
    h_pad=None,
):
    """Plot the variance ratio."""

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    n_components = len(adata.uns["pca"]["variance_ratio"])

    fig = plt.figure(figsize=fig_size)
    if log:
        plt.plot(range(n_components), np.log(adata.uns["pca"]["variance_ratio"]))
    else:
        plt.plot(range(n_components), adata.uns["pca"]["variance_ratio"])
    if show_cutoff:
        n_pcs = adata.uns["pca"]["n_pcs"]
        print(f"the number of selected PC is: {n_pcs}")
        plt.axvline(n_pcs, ls="--", c="red")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=5)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)


def pcs_features(
    adata,
    log=False,
    size=3,
    show_cutoff=True,
    fig_size=(3, 3),
    fig_ncol=3,
    save_fig=None,
    fig_path=None,
    fig_name="qc.pdf",
    pad=1.08,
    w_pad=None,
    h_pad=None,
):
    """Plot features that contribute to the top PCs."""

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    n_pcs = adata.uns["pca"]["n_pcs"]
    n_features = adata.uns["pca"]["PCs"].shape[0]

    fig_nrow = int(np.ceil(n_pcs / fig_ncol))
    fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow))

    for i in range(n_pcs):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
        if log:
            ax_i.scatter(
                range(n_features),
                np.log(
                    np.sort(
                        np.abs(
                            adata.uns["pca"]["PCs"][:, i],
                        )
                    )[::-1]
                ),
                s=size,
            )
        else:
            ax_i.scatter(
                range(n_features),
                np.sort(
                    np.abs(
                        adata.uns["pca"]["PCs"][:, i],
                    )
                )[::-1],
                s=size,
            )
        n_ft_selected_i = len(adata.uns["pca"]["features"][f"pc_{i}"])
        if show_cutoff:
            ax_i.axvline(n_ft_selected_i, ls="--", c="red")
        ax_i.set_xlabel("Feautures")
        ax_i.set_ylabel("Loadings")
        ax_i.locator_params(axis="x", nbins=3)
        ax_i.locator_params(axis="y", nbins=5)
        ax_i.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_i.set_title(f"PC {i}")
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)


def variable_genes(
    adata,
    show_texts=False,
    n_texts=10,
    size=8,
    text_size=10,
    fig_size=(4, 4),
    save_fig=None,
    fig_path=None,
    fig_name="plot_variable_genes.pdf",
    pad=1.08,
    w_pad=None,
    h_pad=None,
):
    """Plot highly variable genes."""

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    means = adata.var["means"]
    variances_norm = adata.var["variances_norm"]
    mask = adata.var["highly_variable"]
    genes = adata.var_names

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(means[~mask], variances_norm[~mask], s=size, c="#1F2433")
    ax.scatter(means[mask], variances_norm[mask], s=size, c="#ce3746")
    ax.set_xscale(value="log")

    if show_texts:
        ids = variances_norm.values.argsort()[-n_texts:][::-1]
        texts = [
            plt.text(
                means[i],
                variances_norm[i],
                genes[i],
                fontdict={
                    "family": "serif",
                    "color": "black",
                    "weight": "normal",
                    "size": text_size,
                },
            )
            for i in ids
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black"))

    ax.set_xlabel("average expression")
    ax.set_ylabel("standardized variance")
    ax.locator_params(axis="x", tight=True)
    ax.locator_params(axis="y", tight=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)


def _scatterplot2d(
    df,
    x,
    y,
    list_hue=None,
    hue_palette=None,
    drawing_order="sorted",
    dict_drawing_order=None,
    size=8,
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=0.8,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name="scatterplot2d.pdf",
    copy=False,
    **kwargs,
):
    """2d scatter plot

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    list_hue: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    list_ax = list()
    if list_hue is None:
        list_hue = [None]
    else:
        for hue in list_hue:
            if hue not in df.columns:
                raise ValueError(f"could not find {hue}")
        if hue_palette is None:
            hue_palette = dict()
        assert isinstance(hue_palette, dict), "`hue_palette` must be dict"
        legend_order = {
            hue: np.unique(df[hue])
            for hue in list_hue
            if (is_string_dtype(df[hue]) or is_categorical_dtype(df[hue]))
        }
        if fig_legend_order is not None:
            if not isinstance(fig_legend_order, dict):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for hue in fig_legend_order.keys():
                if hue in legend_order.keys():
                    legend_order[hue] = fig_legend_order[hue]
                else:
                    print(
                        f"{hue} is ignored for ordering legend labels"
                        "due to incorrect name or data type"
                    )

    if dict_drawing_order is None:
        dict_drawing_order = dict()
    assert drawing_order in [
        "sorted",
        "random",
        "original",
    ], "`drawing_order` must be one of ['original', 'sorted', 'random']"

    if len(list_hue) < fig_ncol:
        fig_ncol = len(list_hue)
    fig_nrow = int(np.ceil(len(list_hue) / fig_ncol))
    fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow))
    for i, hue in enumerate(list_hue):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
        if hue is None:
            sc_i = sns.scatterplot(
                ax=ax_i, x=x, y=y, data=df, alpha=alpha, linewidth=0, s=size, **kwargs
            )
        else:
            if is_string_dtype(df[hue]) or is_categorical_dtype(df[hue]):
                if hue in hue_palette.keys():
                    palette = hue_palette[hue]
                else:
                    palette = None
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == "sorted":
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == "random":
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = sns.scatterplot(
                    ax=ax_i,
                    x=x,
                    y=y,
                    hue=hue,
                    hue_order=legend_order[hue],
                    data=df_updated,
                    alpha=alpha,
                    linewidth=0,
                    palette=palette,
                    s=size,
                    **kwargs,
                )
                ax_i.legend(
                    bbox_to_anchor=(1, 0.5),
                    loc="center left",
                    ncol=fig_legend_ncol,
                    frameon=False,
                )
            else:
                vmin_i = df[hue].min() if vmin is None else vmin
                vmax_i = df[hue].max() if vmax is None else vmax
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == "sorted":
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == "random":
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = ax_i.scatter(
                    df_updated[x],
                    df_updated[y],
                    c=df_updated[hue],
                    vmin=vmin_i,
                    vmax=vmax_i,
                    alpha=alpha,
                    s=size,
                )
                cbar = plt.colorbar(sc_i, ax=ax_i, pad=0.01, fraction=0.05, aspect=40)
                cbar.solids.set_edgecolor("face")
                cbar.ax.locator_params(nbins=5)
        ax_i.set_xlabel(x)
        ax_i.set_ylabel(y)
        ax_i.locator_params(axis="x", nbins=5)
        ax_i.locator_params(axis="y", nbins=5)
        ax_i.tick_params(axis="both", labelbottom=True, labelleft=True)
        ax_i.set_title(hue)
        list_ax.append(ax_i)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)
    if copy:
        return list_ax


def _scatterplot2d_plotly(
    df,
    x,
    y,
    list_hue=None,
    hue_palette=None,
    drawing_order="sorted",
    fig_size=None,
    fig_ncol=3,
    fig_legend_order=None,
    alpha=0.8,
    save_fig=None,
    fig_path=None,
    **kwargs,
):
    """interactive 2d scatter plot by Plotly

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    list_hue: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    for hue in list_hue:
        if hue not in df.columns:
            raise ValueError(f"could not find {hue} in `df.columns`")
    if hue_palette is None:
        hue_palette = dict()
    assert isinstance(hue_palette, dict), "`hue_palette` must be dict"

    assert drawing_order in [
        "sorted",
        "random",
        "original",
    ], "`drawing_order` must be one of ['original', 'sorted', 'random']"

    legend_order = {
        hue: np.unique(df[hue])
        for hue in list_hue
        if (is_string_dtype(df[hue]) or is_categorical_dtype(df[hue]))
    }
    if fig_legend_order is not None:
        if not isinstance(fig_legend_order, dict):
            raise TypeError("`fig_legend_order` must be a dictionary")
        for hue in fig_legend_order.keys():
            if hue in legend_order.keys():
                legend_order[hue] = fig_legend_order[hue]
            else:
                print(
                    f"{hue} is ignored for ordering legend labels"
                    "due to incorrect name or data type"
                )

    if len(list_hue) < fig_ncol:
        fig_ncol = len(list_hue)
    fig_nrow = int(np.ceil(len(list_hue) / fig_ncol))
    fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow))
    for hue in list_hue:
        if hue in hue_palette.keys():
            palette = hue_palette[hue]
        else:
            palette = None
        if drawing_order == "sorted":
            df_updated = df.sort_values(by=hue)
        elif drawing_order == "random":
            df_updated = df.sample(frac=1, random_state=100)
        else:
            df_updated = df
        fig = px.scatter(
            df_updated,
            x=x,
            y=y,
            color=hue,
            opacity=alpha,
            color_continuous_scale=px.colors.sequential.Viridis,
            color_discrete_map=palette,
            **kwargs,
        )
        fig.update_layout(legend={"itemsizing": "constant"}, width=500, height=500)
        fig.show(renderer="notebook")


# TO-DO add 3D plot
def umap(
    adata,
    color=None,
    dict_palette=None,
    n_components=None,
    comp1=1,
    comp2=2,
    comp3=3,
    size=8,
    drawing_order="sorted",
    dict_drawing_order=None,
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=1,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name="scatterplot2d.pdf",
    plotly=False,
    **kwargs,
):
    """Plot coordinates in UMAP

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    color: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    size: `int` (default: 8)
        Point size.
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    if n_components is None:
        n_components = min(3, adata.obsm["X_umap"].shape[1])
    if n_components not in [2, 3]:
        raise ValueError("n_components should be 2 or 3")
    if n_components > adata.obsm["X_umap"].shape[1]:
        print(
            f"`n_components` is greater than the available dimension.\n"
            f"It is corrected to {adata.obsm['X_umap'].shape[1]}"
        )
        n_components = adata.obsm["X_umap"].shape[1]

    if dict_palette is None:
        dict_palette = dict()
    df_plot = pd.DataFrame(
        index=adata.obs.index,
        data=adata.obsm["X_umap"],
        columns=["UMAP" + str(x + 1) for x in range(adata.obsm["X_umap"].shape[1])],
    )
    if color is None:
        _scatterplot2d(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            drawing_order=drawing_order,
            size=size,
            fig_size=fig_size,
            alpha=alpha,
            pad=pad,
            w_pad=w_pad,
            h_pad=h_pad,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_name=fig_name,
            **kwargs,
        )
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if ann in adata.obs_keys():
                df_plot[ann] = adata.obs[ann]
                if not is_numeric_dtype(df_plot[ann]):
                    if "color" not in adata.uns_keys():
                        adata.uns["color"] = dict()

                    if ann not in dict_palette.keys():
                        if (ann + "_color" in adata.uns["color"].keys()) and all(
                            np.isin(
                                np.unique(df_plot[ann]),
                                list(adata.uns["color"][ann + "_color"].keys()),
                            )
                        ):
                            dict_palette[ann] = adata.uns["color"][ann + "_color"]
                        else:
                            dict_palette[ann] = generate_palette(adata.obs[ann])
                            adata.uns["color"][ann + "_color"] = dict_palette[
                                ann
                            ].copy()
                    else:
                        if ann + "_color" not in adata.uns["color"].keys():
                            adata.uns["color"][ann + "_color"] = dict_palette[
                                ann
                            ].copy()

            elif ann in adata.var_names:
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(
                    f"could not find {ann} in `adata.obs.columns`"
                    " and `adata.var_names`"
                )
        if plotly:
            _scatterplot2d_plotly(
                df_plot,
                x="UMAP1",
                y="UMAP2",
                list_hue=color,
                hue_palette=dict_palette,
                drawing_order=drawing_order,
                fig_size=fig_size,
                fig_ncol=fig_ncol,
                fig_legend_order=fig_legend_order,
                alpha=alpha,
                save_fig=save_fig,
                fig_path=fig_path,
                **kwargs,
            )
        else:
            _scatterplot2d(
                df_plot,
                x="UMAP1",
                y="UMAP2",
                list_hue=color,
                hue_palette=dict_palette,
                drawing_order=drawing_order,
                dict_drawing_order=dict_drawing_order,
                size=size,
                fig_size=fig_size,
                fig_ncol=fig_ncol,
                fig_legend_ncol=fig_legend_ncol,
                fig_legend_order=fig_legend_order,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                pad=pad,
                w_pad=w_pad,
                h_pad=h_pad,
                save_fig=save_fig,
                fig_path=fig_path,
                fig_name=fig_name,
                **kwargs,
            )


def graph(
    adata,
    comp1=1,
    comp2=2,
    color=None,
    dict_palette=None,
    size=8,
    drawing_order="random",
    dict_drawing_order=None,
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    alpha=0.9,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name="plot_graph.pdf",
    vmin=None,
    vmax=None,
    show_node=False,
    show_text=False,
    key="epg",
    **kwargs,
):
    """Plot principal graph

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.

    Returns
    -------
    """
    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    if dict_palette is None:
        dict_palette = dict()

    mat_conn = adata.uns[key]["conn"]
    node_pos = adata.uns[key]["node_pos"]
    epg_params = adata.uns[key]["params"]
    obsm = epg_params["obsm"]
    layer = epg_params["layer"]
    G = nx.from_scipy_sparse_matrix(mat_conn)

    if obsm is not None:
        X = adata.obsm[obsm].copy()
    elif layer is not None:
        X = adata.layers[layer].copy()
    else:
        X = adata.X.copy()
    df_plot = pd.DataFrame(
        index=adata.obs.index,
        data=X[:, [comp1 - 1, comp2 - 1]],
        columns=[f"Dim {comp1}", f"Dim {comp2}"],
    )
    if color is None:
        list_ax = _scatterplot2d(
            df_plot,
            x=f"Dim {comp1}",
            y=f"Dim {comp2}",
            drawing_order=drawing_order,
            size=size,
            fig_size=fig_size,
            alpha=alpha,
            pad=pad,
            w_pad=w_pad,
            h_pad=h_pad,
            save_fig=False,
            copy=True,
            **kwargs,
        )
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if ann in adata.obs_keys():
                df_plot[ann] = adata.obs[ann]
                if not is_numeric_dtype(df_plot[ann]):
                    if "color" not in adata.uns_keys():
                        adata.uns["color"] = dict()

                    if ann not in dict_palette.keys():
                        if (ann + "_color" in adata.uns["color"].keys()) and all(
                            np.isin(
                                np.unique(df_plot[ann]),
                                list(adata.uns["color"][ann + "_color"].keys()),
                            )
                        ):
                            dict_palette[ann] = adata.uns["color"][ann + "_color"]
                        else:
                            dict_palette[ann] = generate_palette(adata.obs[ann])
                            adata.uns["color"][ann + "_color"] = dict_palette[
                                ann
                            ].copy()
                    else:
                        if ann + "_color" not in adata.uns["color"].keys():
                            adata.uns["color"][ann + "_color"] = dict_palette[
                                ann
                            ].copy()

            elif ann in adata.var_names:
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(
                    f"could not find {ann} in `adata.obs.columns`"
                    " and `adata.var_names`"
                )
        list_ax = _scatterplot2d(
            df_plot,
            x=f"Dim {comp1}",
            y=f"Dim {comp2}",
            list_hue=color,
            hue_palette=dict_palette,
            drawing_order=drawing_order,
            dict_drawing_order=dict_drawing_order,
            size=size,
            fig_size=fig_size,
            fig_ncol=fig_ncol,
            fig_legend_ncol=fig_legend_ncol,
            fig_legend_order=fig_legend_order,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            pad=pad,
            w_pad=w_pad,
            h_pad=h_pad,
            save_fig=False,
            copy=True,
            **kwargs,
        )

    for ax in list_ax:
        if show_node:
            ax.scatter(node_pos[:, comp1 - 1], node_pos[:, comp2 - 1], c="black")
        if show_text:
            for i in np.arange(node_pos.shape[0]):
                ax.text(
                    node_pos[i, comp1 - 1],
                    node_pos[i, comp2 - 1],
                    i,
                    color="black",
                    ha="left",
                    va="bottom",
                )
        for edge_i in G.edges():
            ax.plot(node_pos[edge_i, comp1 - 1], node_pos[edge_i, comp2 - 1], c="black")
    if save_fig:
        fig = plt.gcf()
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)


def plot_features_in_pseudotime(
    adatas,
    assays,
    features,
    source=None,
    target=None,
    nodes_to_include=None,
    frac=0.2,
    key="epg",
):
    fig = make_subplots(rows=len(features), cols=len(adatas), subplot_titles=assays)

    for i in range(len(adatas)):
        df, path_alias = _utils.get_expdata(
            adatas[i], source, target, nodes_to_include, key
        )
        for j in range(len(features)):
            feature = features[j]
            fig.add_trace(
                go.Scatter(
                    x=df[f"{key}_pseudotime"],
                    y=df[feature],
                    opacity=0.8,
                    mode="markers",
                    name=assays[i],
                ),
                row=j + 1,
                col=i + 1,
            )

            y_lowess = lowess(df[feature], df[f"{key}_pseudotime"], frac=frac)
            fig.add_trace(
                go.Scatter(
                    x=y_lowess[:, 0],
                    y=y_lowess[:, 1],
                    name=" ",
                    line=dict(color="black"),
                ),
                row=j + 1,
                col=i + 1,
            )
            fig.update_xaxes(title_text="pseudotime", row=j + 1, col=i + 1)
            fig.update_yaxes(title_text=str(feature), row=j + 1, col=i + 1)

    fig.update_layout(
        dict(height=400, width=1000, plot_bgcolor="white", title_text=str(path_alias))
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="lightgrey",
        showline=True,
        linewidth=1,
        linecolor="black",
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="lightgrey",
        showline=True,
        linewidth=1,
        linecolor="black",
    )
    fig.update_traces(marker=dict(size=3))
    return fig
