[![CI](https://github.com/pinellolab/stream2/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/stream2/actions/workflows/CI.yml)

# STREAM2 (Latest version v0.1)

STREAM2: Fast, scalable, and interactive trajectory analysis of single-cell omics data

![simba](./docs/source/_static/img/logo.png?raw=true)

Introduction
------------

STREAM2 (**S**ingle-cell **T**rajectories **R**econstruction, **E**xploration **A**nd **M**apping) is an interactive pipeline capable of disentangling and visualizing complex trajectories from for single-cell omics data.

The four key innovations of STREAM2 are: 
1) STREAM2 can learn more biologically meaningful trajectories in a semi-supervised way by leveraging external information (e.g. time points, FACS labels, predefined relations of clusters, etc.); 
2) STREAM2 is able to learn not only linear or tree-like structures but also more complex graphs with loops or disconnected components; 
3) STREAM2 supports trajectory inference for various single-cell assays such as gene expression, chromatin accessibility, protein expression level, and DNA methylation; 
4) STREAM2 introduces a flexible and powerful path-based marker detection procedure. In addition, we provide a scalable and fast python package along with a comprehensive documentation website to facilitate STREAM2 analysis. 
We also provide an accompanying interactive website to allow non-expert users to freely and easily compute and explore the obtained trajectories.

![simba](./docs/source/_static/img/Fig1_V2.1.jpg?raw=true)

Installation
------------
```sh
$ pip install git+https://github.com/pinellolab/STREAM2
```

Tutorials
--------
Preliminary tutorials for the usage of STREAM2 can be found at **STREAM2_tutorials** repository [https://github.com/pinellolab/STREAM2_tutorials]. 
