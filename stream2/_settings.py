"""Configuration for STREAM2"""

import os
import seaborn as sns
import matplotlib as mpl


class stream2Config:
    """configuration class for STREAM2"""

    def __init__(self,
                 workdir='./result_stream2',
                 save_fig=False,
                 n_jobs=1):
        self.workdir = workdir
        self.save_fig = save_fig
        self.n_jobs = n_jobs

    def set_figure_params(self,
                          context='notebook',
                          style='white',
                          palette='deep',
                          font='sans-serif',
                          font_scale=1.1,
                          color_codes=True,
                          save_fig=False,
                          dpi=80,
                          dpi_save=150,
                          fig_size=[5.4, 4.8],
                          rc=None):
        """ Set global parameters for figures. Modified from sns.set()
        Parameters
        ----------
        context : string or dict
            Plotting context parameters, see seaborn :func:`plotting_context
        style: `string`,optional (default: 'white')
            Axes style parameters, see seaborn :func:`axes_style`
        palette : string or sequence
            Color palette, see seaborn :func:`color_palette`
        font_scale: `float`, optional (default: 1.3)
            Separate scaling factor to independently
            scale the size of the font elements.
        color_codes : `bool`, optional (default: True)
            If ``True`` and ``palette`` is a seaborn palette,
            remap the shorthand color codes (e.g. "b", "g", "r", etc.)
            to the colors from this palette.
        dpi: `int`,optional (default: 80)
            Resolution of rendered figures.
        dpi_save: `int`,optional (default: 150)
            Resolution of saved figures.
        rc: `dict`,optional (default: None)
            rc settings properties.
            Parameter mappings to override the values in the preset style.
            Please see https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
        """
        # mpl.rcParams.update(mpl.rcParamsDefault)
        sns.set(context=context,
                style=style,
                palette=palette,
                font=font,
                font_scale=font_scale,
                color_codes=color_codes,
                rc={'figure.dpi': dpi,
                    'savefig.dpi': dpi_save,
                    'figure.figsize': fig_size,
                    'image.cmap': 'viridis',
                    'lines.markersize': 6,
                    'legend.columnspacing': 0.1,
                    'legend.borderaxespad': 0.1,
                    'legend.handletextpad': 0.1,
                    'pdf.fonttype': 42,
                    })
        if(rc is not None):
            assert isinstance(rc, dict), "rc must be dict"
            for key, value in rc.items():
                if key in mpl.rcParams.keys():
                    mpl.rcParams[key] = value
                else:
                    raise Exception("unrecognized property '%s'" % key)

    def set_workdir(self, workdir=None):
        """Set working directory.

        Parameters
        ----------
        workdir: `str`, optional (default: None)
            Working directory.

        Returns
        -------
        """
        if(workdir is None):
            workdir = self.workdir
            print("Using default working directory.")
        if(not os.path.exists(workdir)):
            os.makedirs(workdir)
        self.workdir = workdir
        print('Saving results in: %s' % workdir)


settings = stream2Config()
