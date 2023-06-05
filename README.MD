# Evaluating Self-Supervised Graph Neural Networks using GraphWorld

This repository contains the code and accompanying materials for our master's thesis titled "_Evaluating Self-Supervised Graph Neural Networks using GraphWorld_" at the IT-University of Copenhagen by Daniel Enggaard and Bertram Kehler. 


# Acknowledgement
We have forked this repository from the original [GraphWorld repository]([link-to-google-repo](https://github.com/google-research/graphworld)), developed by Google and authored by Palowitch et al. The GraphWorld framework provides a solid foundation for our work, and we have utilized and modified it to suit the requirements of our master's thesis. Please refer to the [original GraphWorld repository](https://github.com/google-research/graphworld) for more details about the framework and its usage.


# Folder structure

````bash
.
├── configs
│   ├── ...
│   └── SSL_nodeclassification
│       ├── mode_1                  # Configuration files for mode 1
│       │   └── SSL_methods         # Configuration files for all SSL methods in mode 1 experiments
│       ├── mode_3                  # Configuration files for mode 2/3
│       │   └── SSL_methods         # Configuration files for all SSL methods in mode 2/3 experiments
│       ├── preliminary             # Configuration files for preliminary experiments
│       │   └── SSL_methods         # Configuration files for all SSL methods in the preliminary experiments
│       └── sbm
├── graph_world
│   └── self_supervised_learning    # Directory for graph SSL-related code
│       ├── ...
│       └── pretext_tasks           # Graph SSL implementations
````

