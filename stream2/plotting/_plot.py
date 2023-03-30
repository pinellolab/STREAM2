"""plotting functions."""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
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

from slugify import slugify

from .._settings import settings
from ._utils import generate_palette
from .. import _utils
from ._utils_stream import (
    _check_is_tree,
    _add_stream_sc_pos,
    _cal_stream_polygon_string,
    _cal_stream_polygon_numeric
)


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
    """Violin plot."""

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
                ax=ax_i,
                y=obs,
                data=df_plot,
                color="black",
                jitter=jitter,
                s=size,
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
                os.path.join(fig_path, fig_name),
                pad_inches=1,
                bbox_inches="tight",
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
                ax=ax_i,
                y=var,
                data=df_plot,
                color="black",
                jitter=jitter,
                s=size,
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
                os.path.join(fig_path, fig_name),
                pad_inches=1,
                bbox_inches="tight",
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
    fig_name="plot_hist.pdf",
    **kwargs,
):
    """histogram plot."""

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
                os.path.join(fig_path, fig_name),
                pad_inches=1,
                bbox_inches="tight",
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
                os.path.join(fig_path, fig_name),
                pad_inches=1,
                bbox_inches="tight",
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
        plt.plot(
            range(n_components), np.log(adata.uns["pca"]["variance_ratio"])
        )
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
        plt.savefig(
            os.path.join(fig_path, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
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
    fig = plt.figure(
        figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
    )

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
        plt.savefig(
            os.path.join(fig_path, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
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
        fig.savefig(
            os.path.join(fig_path, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
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
    cbar_pad=0.01,
    cbar_fraction=0.05,
    cbar_aspect=40,
    save_fig=None,
    fig_path=None,
    fig_name="scatterplot2d.pdf",
    copy=False,
    **kwargs,
):
    """2d scatter plot.

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
    fig = plt.figure(
        figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
    )
    for i, hue in enumerate(list_hue):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)
        if hue is None:
            sc_i = sns.scatterplot(
                ax=ax_i,
                x=x,
                y=y,
                data=df,
                alpha=alpha,
                linewidth=0,
                s=size,
                **kwargs,
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
                cbar = plt.colorbar(
                    sc_i,
                    ax=ax_i,
                    pad=cbar_pad,
                    fraction=cbar_fraction,
                    aspect=cbar_aspect
                )
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
        plt.savefig(
            os.path.join(fig_path, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
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
    """interactive 2d scatter plot by Plotly.

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
    fig = plt.figure(
        figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow)
    )
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
        fig.update_layout(
            legend={"itemsizing": "constant"}, width=500, height=500
        )
        fig.show(renderer="notebook")


# TO-DO add 3D plot
def dimension_reduction(
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
    cbar_pad=0.01,
    cbar_fraction=0.05,
    cbar_aspect=40,
    save_fig=None,
    fig_path=None,
    fig_name="scatterplot2d.pdf",
    plotly=False,
    **kwargs,
):
    """Plot coordinates in low dimensions.

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
        n_components = min(3, adata.obsm["X_dr"].shape[1])
    if n_components not in [2, 3]:
        raise ValueError("n_components should be 2 or 3")
    if n_components > adata.obsm["X_dr"].shape[1]:
        print(
            f"`n_components` is greater than the available dimension.\n"
            f"It is corrected to {adata.obsm['X_dr'].shape[1]}"
        )
        n_components = adata.obsm["X_dr"].shape[1]

    if dict_palette is None:
        dict_palette = dict()
    df_plot = pd.DataFrame(
        index=adata.obs.index,
        data=adata.obsm["X_dr"],
        columns=[
            "Dim_" + str(x + 1) for x in range(adata.obsm[
                                                       "X_dr"].shape[1])
        ],
    )
    if color is None:
        _scatterplot2d(
            df_plot,
            x="Dim_1",
            y="Dim_2",
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
                    if "color" not in adata.uns.keys():
                        adata.uns["color"] = dict()

                    if ann not in dict_palette.keys():
                        if (
                            ann + "_color" in adata.uns["color"].keys()
                        ) and all(
                            np.isin(
                                np.unique(df_plot[ann]),
                                list(
                                    adata.uns["color"][ann + "_color"].keys()
                                ),
                            )
                        ):
                            dict_palette[ann] = adata.uns["color"][
                                ann + "_color"
                            ]
                        else:
                            dict_palette[ann] = generate_palette(
                                adata.obs[ann]
                            )
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
                x="Dim_1",
                y="Dim_2",
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
                x="Dim_1",
                y="Dim_2",
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
                cbar_pad=cbar_pad,
                cbar_fraction=cbar_fraction,
                cbar_aspect=cbar_aspect,
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
    """Plot principal graph.

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
    G = nx.from_scipy_sparse_array(mat_conn)

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
                    if "color" not in adata.uns.keys():
                        adata.uns["color"] = dict()

                    if ann not in dict_palette.keys():
                        if (
                            ann + "_color" in adata.uns["color"].keys()
                        ) and all(
                            np.isin(
                                np.unique(df_plot[ann]),
                                list(
                                    adata.uns["color"][ann + "_color"].keys()
                                ),
                            )
                        ):
                            dict_palette[ann] = adata.uns["color"][
                                ann + "_color"
                            ]
                        else:
                            dict_palette[ann] = generate_palette(
                                adata.obs[ann]
                            )
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
            ax.scatter(
                node_pos[:, comp1 - 1], node_pos[:, comp2 - 1], c="black"
            )
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
            ax.plot(
                node_pos[edge_i, comp1 - 1],
                node_pos[edge_i, comp2 - 1],
                c="black",
            )
    if save_fig:
        fig = plt.gcf()
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(
            os.path.join(fig_path, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
        plt.close(fig)


def feature_path(
    adatas,
    features,
    source=None,
    target=None,
    nodes_to_include=None,
    assays=None,
    frac=0.2,
    height=400,
    width=400,
    key="epg",
    copy=False,
):
    lowess = sm.nonparametric.lowess
    fig = make_subplots(
        rows=len(features), cols=len(adatas), subplot_titles=assays
    )

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
        dict(
            height=height,
            width=width,
            plot_bgcolor="white",
            title_text=str(path_alias),
        )
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
    if copy:
        return fig


def stream_sc(
    adata,
    source=0,
    key='epg',
    color=None,
    dict_palette=None,
    dist_scale=1,
    dist_pctl=95,
    size=8,
    drawing_order="sorted",
    dict_drawing_order=None,
    preference=None,
    fig_size=(7, 4.5),
    fig_ncol=1,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=0.8,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    cbar_pad=0.04,
    cbar_fraction=0.05,
    cbar_aspect=40,
    show_text=True,
    show_graph=True,
    save_fig=False,
    fig_path=None,
    fig_name='plot_stream_sc.png',
    **kwargs
):
    """Generate stream plot at single cell level (aka, subway map plots)

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'S0'):
        The starting node
    color: `list` optional (default: None)
        Column names of observations (adata.obs.columns) or
        variable names(adata.var_names). A list of names to be plotted.
    dist_scale: `float`,optional (default: 1)
        Scaling factor to scale the distance from cells to tree branches
        (by default, it keeps the same distance as in original manifold)
    dist_pctl: `int`, optional (default: 95)
        Percentile of cells' distances from branches (between 0 and 100)
        used for calculating the distances between branches.
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes
        are prioritized and put on the top part of stream plot.
        The higher ranks the node have,
        the closer to the top the branch with the node is.
    fig_size: `tuple`, optional (default: (7,4.5))
        figure size.
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for ategorical variable.
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
    show_text: `bool`, optional (default: False)
        If True, node state label will be shown
    show_graph: `bool`, optional (default: False)
        If True, the learnt principal graph will be shown
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        if save_fig is True, specify figure path.
        if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.
    plotly: `bool`, optional (default: False)
        if True, plotly will be used to make interactive plots

    Returns
    -------
    updates `adata` with the following fields.
    stream_tree: `dict` (`adata.uns['stream_tree']`)
        Store details of the tree structure used in stream plots.
    """

    assert _check_is_tree(adata, key=key), \
        "`.pl.stream_sc()` only works for a tree structure"
    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    if dict_palette is None:
        dict_palette = dict()

    _add_stream_sc_pos(
        adata,
        source=source,
        dist_scale=dist_scale,
        dist_pctl=dist_pctl,
        preference=preference,
        key=key,
    )
    stream_node = adata.uns['stream_tree']['node']
    stream_node_pos = adata.uns['stream_tree']['node_pos']
    stream_edge_pos = adata.uns['stream_tree']['edge_pos']

    df_plot = pd.DataFrame(
        index=adata.obs.index,
        data=adata.uns['stream_tree']['cell_pos'],
        columns=['pseudotime', 'dist'])

    if color is None:
        list_ax = _scatterplot2d(
                    df_plot,
                    x="pseudotime",
                    y="dist",
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
                    if "color" not in adata.uns.keys():
                        adata.uns["color"] = dict()

                    if ann not in dict_palette.keys():
                        if (
                            ann + "_color" in adata.uns["color"].keys()
                        ) and all(
                            np.isin(
                                np.unique(df_plot[ann]),
                                list(
                                    adata.uns["color"][ann + "_color"].keys()
                                ),
                            )
                        ):
                            dict_palette[ann] = adata.uns["color"][
                                ann + "_color"
                            ]
                        else:
                            dict_palette[ann] = generate_palette(
                                adata.obs[ann]
                            )
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
                    x="pseudotime",
                    y="dist",
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
                    cbar_pad=cbar_pad,
                    cbar_fraction=cbar_fraction,
                    cbar_aspect=cbar_aspect,
                    save_fig=False,
                    copy=True,
                    **kwargs,
        )
    for ax_i in list_ax:
        ax_i.set_xlabel("pseudotime", labelpad=2)
        ax_i.spines['left'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['top'].set_visible(False)
        ax_i.get_yaxis().set_visible(False)
        ax_i.locator_params(axis='x', nbins=8)
        ax_i.tick_params(axis="x", pad=-1)
        ax_i.plot((1), (0), ls="", marker=">", ms=10, color="k",
                  transform=ax_i.transAxes, clip_on=False)
    if show_graph:
        for ax_i in list_ax:
            for edge_i in stream_edge_pos.keys():
                branch_i_pos = stream_edge_pos[edge_i]
                branch_i = pd.DataFrame(
                    branch_i_pos, columns=range(branch_i_pos.shape[1]))
                for ii in np.arange(
                        start=0, stop=branch_i.shape[0], step=2):
                    if branch_i.iloc[ii, 0] == branch_i.iloc[ii+1, 0]:
                        ax_i.plot(
                            branch_i.iloc[[ii, ii+1], 0],
                            branch_i.iloc[[ii, ii+1], 1],
                            c='#767070',
                            zorder=-1,
                            alpha=0.8)
                    else:
                        ax_i.plot(
                            branch_i.iloc[[ii, ii+1], 0],
                            branch_i.iloc[[ii, ii+1], 1],
                            c='black',
                            alpha=0.8,
                            zorder=-1,)
    if show_text:
        for ax_i in list_ax:
            for i, node_i in enumerate(stream_node):
                ax_i.text(
                    stream_node_pos[i, 0],
                    stream_node_pos[i, 1],
                    node_i,
                    color='black',
                    fontsize=0.9*mpl.rcParams['font.size'],
                    ha='left',
                    va='bottom')
    if save_fig:
        file_path_S = os.path.join(fig_path, f'source_{source}')
        if not os.path.exists(file_path_S):
            os.makedirs(file_path_S)
        plt.savefig(
            os.path.join(file_path_S, fig_name),
            pad_inches=1,
            bbox_inches="tight",
        )
        plt.close()


def stream(
    adata,
    source=0,
    key='epg',
    color=None,
    dict_palette=None,
    preference=None,
    dist_scale=0.9,
    factor_num_win=10,
    factor_min_win=2.0,
    factor_width=2.5,
    factor_nrow=200,
    factor_ncol=400,
    log_scale=False,
    factor_zoomin=100.0,
    fig_size=(7, 4.5),
    fig_legend_order=None,
    fig_legend_ncol=1,
    fig_colorbar_aspect=30,
    vmin=None,
    vmax=None,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=False,
    fig_path=None,
    fig_format='png'
):
    """Generate stream plot at density level

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'S0'):
        The starting node
    color: `list` optional (default: None)
        Column names of observations (adata.obs.columns)
        or variable names(adata.var_names)
         A list of names to be plotted.
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes are preferred
        and will be put on the upper part of stream plot.
        The higher ranks the node have,
        the closer to the top the branch containing the node is.
    dist_scale: `float`,optional (default: 0.9)
        Scaling factor. It controls the width of STREAM plot branches.
        The smaller, the thinner the branch will be.
    factor_num_win: `int`, optional (default: 10)
        Number of sliding windows used for making stream plot.
        It controls the smoothness of STREAM plot.
    factor_min_win: `float`, optional (default: 2.0)
        The minimum number of sliding windows.
        It controls the resolution of STREAM plot.
        The window size is calculated based on shortest branch.
        (suggested range: 1.5~3.0)
    factor_width: `float`, optional (default: 2.5)
        The ratio between length and width of stream plot.
    factor_nrow: `int`, optional (default: 200)
        The number of rows in the array used to plot continuous values.
    factor_ncol: `int`, optional (default: 400)
        The number of columns in the array used to plot continuous values
    log_scale: `bool`, optional (default: False)
        If True,the number of cells (the width) is logarithmized
        when drawing stream plot.
    factor_zoomin: `float`, optional (default: 100.0)
        If log_scale is True, the factor used to zoom in the thin branches
    fig_size: `tuple`, optional (default: (7,4.5))
        figure size.
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for ategorical variable.
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        if save_fig is True, specify figure path.
        if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.

    Returns
    -------
    None

    """

    assert _check_is_tree(adata, key=key), \
        "`.pl.stream()` only works for a tree structure"
    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")

    if source not in adata.uns['stream_tree']['node']:
        raise ValueError(f"There is no source {source}")

    if dict_palette is None:
        dict_palette = dict()

    dict_ann = dict()
    if color is None:
        dict_ann['label'] = pd.Series(index=adata.obs_names, data='unknown')
        legend_order = {'label': ['unknown']}
    else:
        color = [color] if isinstance(color, str) else color
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if ann in adata.obs.columns:
                dict_ann[ann] = adata.obs[ann]
            elif ann in adata.var_names:
                dict_ann[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in "
                                 "`adata.obs.columns` and `adata.var_names`")
        legend_order = {ann: np.unique(dict_ann[ann]) for ann in color
                        if not is_numeric_dtype(dict_ann[ann])}
        if fig_legend_order is not None:
            if not isinstance(fig_legend_order, dict):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for ann in fig_legend_order.keys():
                if ann in legend_order.keys():
                    legend_order[ann] = fig_legend_order[ann]
                else:
                    print(f"{ann} is ignored for ordering legend labels"
                          "due to incorrect name or data type")

    dict_plot = dict()

    list_string_type = [k for k, v in dict_ann.items()
                        if not is_numeric_dtype(v)]
    if len(list_string_type) > 0:
        dict_verts, dict_extent = _cal_stream_polygon_string(
            adata,
            dict_ann,
            source=source,
            preference=preference,
            dist_scale=dist_scale,
            factor_num_win=factor_num_win,
            factor_min_win=factor_min_win,
            factor_width=factor_width,
            log_scale=log_scale,
            factor_zoomin=factor_zoomin,
            key=key)
        dict_plot['string'] = [dict_verts, dict_extent]

    list_numeric_type = [k for k, v in dict_ann.items() if is_numeric_dtype(v)]
    if len(list_numeric_type) > 0:
        verts, extent, ann_order, dict_ann_df, dict_im_array = \
            _cal_stream_polygon_numeric(
                adata,
                dict_ann,
                source=source,
                preference=preference,
                dist_scale=dist_scale,
                factor_num_win=factor_num_win,
                factor_min_win=factor_min_win,
                factor_width=factor_width,
                factor_nrow=factor_nrow,
                factor_ncol=factor_ncol,
                log_scale=log_scale,
                factor_zoomin=factor_zoomin,
                key=key)
        dict_plot['numeric'] = [
            verts, extent, ann_order, dict_ann_df, dict_im_array]

    for ann in dict_ann.keys():
        if not is_numeric_dtype(dict_ann[ann]):
            if "color" not in adata.uns.keys():
                adata.uns["color"] = dict()
            if ann not in dict_palette.keys():
                if (ann + "_color" in adata.uns["color"].keys())\
                    and all(np.isin(
                        np.unique(dict_ann[ann]),
                        list(adata.uns["color"][ann + "_color"].keys()),)):
                    dict_palette[ann] = adata.uns["color"][ann + "_color"]
                else:
                    dict_palette[ann] = generate_palette(dict_ann[ann])
                    if color is not None:
                        adata.uns["color"][ann + "_color"] = \
                            dict_palette[ann].copy()
            verts = dict_plot['string'][0][ann]
            extent = dict_plot['string'][1][ann]
            xmin = extent['xmin']
            xmax = extent['xmax']
            ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
            ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1

            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)
            legend_labels = []
            for ann_i in legend_order[ann]:
                legend_labels.append(ann_i)
                verts_cell = verts[ann_i]
                polygon = Polygon(
                    verts_cell, closed=True,
                    color=dict_palette[ann][ann_i],
                    alpha=0.8, lw=0)
                ax.add_patch(polygon)
            if color is not None:
                ax.legend(
                    legend_labels,
                    bbox_to_anchor=(1.03, 0.5),
                    loc='center left',
                    ncol=fig_legend_ncol,
                    frameon=False,
                    columnspacing=0.4,
                    borderaxespad=0.2,
                    handletextpad=0.3,)
        else:
            verts = dict_plot['numeric'][0]
            extent = dict_plot['numeric'][1]
            ann_order = dict_plot['numeric'][2]
            dict_ann_df = dict_plot['numeric'][3]
            dict_im_array = dict_plot['numeric'][4]
            xmin = extent['xmin']
            xmax = extent['xmax']
            ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
            ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1

            # clip parts according to determined polygon
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)
            for ann_i in ann_order:
                vmin_i = dict_ann_df[ann].loc[ann_i, :].min()\
                    if vmin is None else vmin
                vmax_i = dict_ann_df[ann].loc[ann_i, :].max()\
                    if vmax is None else vmax
                im = ax.imshow(
                    dict_im_array[ann][ann_i],
                    interpolation='bicubic',
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=vmin_i,
                    vmax=vmax_i,
                    aspect='auto')
                verts_cell = verts[ann_i]
                clip_path = Polygon(
                    verts_cell, facecolor='none',
                    edgecolor='none', closed=True)
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)
                cbar = plt.colorbar(
                    im, ax=ax, pad=0.04, fraction=0.02,
                    aspect=fig_colorbar_aspect)
                cbar.ax.locator_params(nbins=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("pseudotime", labelpad=2)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.locator_params(axis='x', nbins=8)
        ax.tick_params(axis="x", pad=-1)
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.transAxes, clip_on=False)
        if color is not None:
            ax.set_title(ann)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if save_fig:
            file_path_S = os.path.join(fig_path, f'source_{source}')
            if not os.path.exists(file_path_S):
                os.makedirs(file_path_S)
            if color is None:
                plt.savefig(os.path.join(
                    file_path_S, 'plot_stream.' + fig_format),
                    pad_inches=1, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(
                    file_path_S,
                    'plot_stream_' + slugify(ann) + '.' + fig_format),
                    pad_inches=1, bbox_inches='tight')
            plt.close(fig)
