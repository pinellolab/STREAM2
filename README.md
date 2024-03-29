[![CI](https://github.com/pinellolab/stream2/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/stream2/actions/workflows/CI.yml)

![simba](./docs/source/_static/img/logo.png?raw=true)

# STREAM2
STREAM2 (**S**ingle-cell **T**rajectories **R**econstruction, **E**xploration **A**nd **M**apping) is an interactive pipeline capable of disentangling and visualizing complex trajectories from for single-cell omics data.


Installation
------------
```sh
$ pip install git+https://github.com/pinellolab/STREAM2
```

Tutorials
---------
Preliminary tutorials for the usage of STREAM2 can be found at **STREAM2_tutorials** repository https://github.com/pinellolab/STREAM2_tutorials. 


Description
-----------
The four key innovations of STREAM2 are: 
1) STREAM2 can learn more biologically meaningful trajectories in a semi-supervised way by leveraging external information (e.g. time points, FACS labels, predefined relations of clusters, etc.); 
2) STREAM2 is able to learn not only linear or tree-like structures but also more complex graphs with loops or disconnected components; 
3) STREAM2 supports trajectory inference for various single-cell assays such as gene expression, chromatin accessibility, protein expression level, and DNA methylation; 
4) STREAM2 introduces a flexible path-based marker detection procedure. In addition, we provide a scalable and fast python package along with a comprehensive documentation website to facilitate STREAM2 analysis. 

![simba](./docs/source/_static/img/Fig1_V2.1.jpg?raw=true)
