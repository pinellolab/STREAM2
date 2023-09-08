.. automodule:: stream2

API
===

Import stream2 as::

   import stream2 as st2

Configuration for STREAM2
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   settings.set_figure_params
   settings.set_workdir


Reading
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   read_csv
   read_h5ad
   read_10X_output
   read_mtx


See more at `anndata <https://anndata.readthedocs.io/en/latest/api.html#reading>`_


Preprocessing
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pp.log_transform
   pp.normalize
   pp.cal_qc_rna
   pp.filter_genes
   pp.pca
   pp.select_variable_genes   


Tools
~~~~~
.. autosummary::
   :toctree: _autosummary

   tl.dimension_reduction
   tl.seed_graph
   tl.learn_graph
   tl.infer_pseudotime
   tl.add_path
   tl.del_path
   tl.get_weights
   tl.extend_leaves
   tl.refit_graph
   tl.project_graph


Plotting
~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pl.pca_variance_ratio
   pl.variable_genes
   pl.violin
   pl.graph
   pl.dimension_reduction
   pl.stream_sc
   pl.stream