# Evaluating Self-Supervised Graph Neural Networks using GraphWorld

This repository contains the code and accompanying materials for our master's thesis titled "_Evaluating Self-Supervised Graph Neural Networks using GraphWorld_" at the IT-University of Copenhagen by Daniel Enggaard and Bertram Kehler.
The results and logging data is not included in this GitHub, but we are happy to provide the data upon request should it benefit future research.

# GraphWorld
We have forked this repository from the original [GraphWorld repository]([link-to-google-repo](https://github.com/google-research/graphworld)), developed by Google and authored by Palowitch et al. The GraphWorld framework provides a solid foundation for our work, and we have utilized and modified it to suit the requirements of our master's thesis. Please refer to the [original GraphWorld repository](https://github.com/google-research/graphworld) for more details about the framework and its usage.


# Folder structure

````bash
.
├── evaluation
│   ├── marg2_plots                     # Animations for varying two properties
│   ├── results                         # Logging files for all results (git-ignored)
│   └── ...                             # Files containing all analyses/plots
├── hpc                                 # Files for running GraphWorld on HPC
├── sripts                              # Scripts for building containers and running GraphWorld
├── src                                 # GraphWorld repository with our modifications
│   ├── configs                         # Gin configuration files for running GraphWorld
│   │   ├── ...                         # Unedited GraphWorld configuration files
│   │   └── SSL_nodeclassification      # Our configuration files                     
│   │       ├── mode_1                  # Configuration files for mode 1
│   │       ├── mode_3                  # Configuration files for mode 2/3
│   │       ├── preliminary             # Configuration files for preliminary experiments
│   │       ├── sbm                     # Configuration files for ADC-SBM generator
│   │       ├── nodeclassification.gin  # Main wrapper for all configuration files
│   │       └── ...                     # Other wrappers for quick experimentation
│   ├── graph_world                     # The codes being executed in a GraphWorld pipeline
│   │   ├── ...                         # Unedited GraphWorld code
│   │   ├── models                      # SSL GNN encoders / non-SSL baselines
│   │   └── SSL_nodeclassification      # SSL-related codes / modifications
│   │       ├── pretext_tasks           # Contains all implementations of the graph SSL methods
│   │       └── ...                     # Utils + Modifications to the GraphWorld pipeline
│   ├── hparam_analysis                 # GraphWorld's own hparam analysis (not used by us)
│   └── notebooks                       # GraphWorld's own notebooks (not used by us)
````
# References
All SSL method implementations are based on the authors' papers and codes, if available. The following list contains an overview of links to these resources. All methods have been adjusted to be compatible with any encoder and training scheme.

| SSL Method | Paper | Code |
|------------|-----------|-----------|
| AttributeMask (AM)   | [Jin et al. 2020](https://arxiv.org/abs/2006.10141) | [SelfTask-GNN](https://github.com/ChandlerBang/SelfTask-GNN) |
| CorruptedFeaturesReconstruction (CFR)   | [Manessi et al. 2021](https://arxiv.org/abs/2011.07267) | None |
| CorruptedEmbeddingsReconstruction (CER)   | [Manessi et al. 2021](https://arxiv.org/abs/2011.07267) | None |
| Autoencoding (AE)   | [Manessi et al. 2021](https://arxiv.org/abs/2011.07267) | None |
| EdgeMask (EM)   | [Jin et al. 2020](https://arxiv.org/abs/2006.10141) | [SelfTask-GNN](https://github.com/ChandlerBang/SelfTask-GNN) |
| Graph Auto-Encoder (GAE)   | [Kipf and Welling 2016](https://arxiv.org/abs/1611.07308) | [Graph Auto-Encoders](https://github.com/tkipf/gae) |
| PairwiseAttrSim (PAS)   | [Jin et al. 2020](https://arxiv.org/abs/2006.10141) | [SelfTask-GNN](https://github.com/ChandlerBang/SelfTask-GNN) |
| S<sup>2</sup>GRL (S2GRL)   | [Peng et al. 2020](https://arxiv.org/abs/2003.01604) | None |
| GraphPartitioning (GP)   | [You et al. 2020](https://arxiv.org/abs/2006.09136) | [SS-GCNs](https://github.com/Shen-Lab/SS-GCNs) |
| Node Clustering With Alignment (NC)   | [Sun et al. 2020](https://arxiv.org/abs/1902.11038) | [M3S](https://github.com/datake/M3S) |
| Graph Contrastive representation learning with Adaptive augmentation (GCA)   | [Zhu et al. 2021](https://arxiv.org/abs/2010.14945) | [GCA](https://github.com/CRIPAC-DIG/GCA) |
| Bootstrapped Graph Latents (BGRL)   | [Thakoor et al. 2021](https://arxiv.org/abs/2102.06514) | [BGRL_Pytorch](https://github.com/Namkyeong/BGRL_Pytorch) |
| SelfGNNPPR (S-PPR)   | [Kefato and Girdzijauskas 2021](https://arxiv.org/abs/2103.14958) | [SelfGNN](https://github.com/zekarias-tilahun/SelfGNN) |
| SelfGNNSplit (S-Split)   | [Kefato and Girdzijauskas 2021](https://arxiv.org/abs/2103.14958) | [SelfGNN](https://github.com/zekarias-tilahun/SelfGNN) |
| Multi-scale contrastive siamese network (MERIT)   | [Jin et al. 2021](https://arxiv.org/abs/2105.05682) | [MERIT](https://github.com/GRAND-Lab/MERIT) |
| Graph Barlow Twins (G-BT)  | [Bielak et al. 2022](https://arxiv.org/abs/2106.02466) | [Graph Barlow Twins](https://github.com/pbielak/graph-barlow-twins) |
| Deep Graph Infomax (DGI)  | [Hjelm et al. 2018](https://arxiv.org/abs/1808.06670) | [DGI](https://github.com/PetarV-/DGI) |
| Graph InfoClust (GIC)  | [Mavromatis and Karypis 2020](https://arxiv.org/abs/2009.06946) | [Graph-InfoClust-GIC](https://github.com/cmavro/Graph-InfoClust-GIC) |
| Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning (SUBG-CON)  | [Jiao et al. 2020](https://arxiv.org/abs/2009.10273) | [Subg-Con](https://github.com/yzjiao/Subg-Con) |
| Manessi et al. (MEtAl)  | [Manessi et al. 2021](https://arxiv.org/abs/2011.07267) | None |
| Graph Contrastive Adjusted Zooming (G-Zoom) | [Zheng et al. 2022](https://arxiv.org/abs/2111.10698) | None |
| Multi-View Mutual Information Feature-Topology (MVMI-FT) | [Fan et al. 2022](https://arxiv.org/abs/2105.06715) | [MaxMIAcrossFT](https://github.com/xiaolongo/MaxMIAcrossFT) |


<br><br>
The other two main resources for codes are GraphWorld and Pytorch Geometric:
| Resource | Paper | Code |
|------------|-----------|-----------|
| GraphWorld   | [Palowitch et al. 2022](https://arxiv.org/abs/2203.00112) | [GraphWorld](https://github.com/google-research/graphworld) |
| Pytorch Geometric (PyG)   | [Fey and Lenssen 2019](https://arxiv.org/abs/1903.02428) | [PyG](https://github.com/pyg-team/pytorch_geometric) |