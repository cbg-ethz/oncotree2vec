<div align="left">
<img src="./docs/logo.png" width="300" height="auto">
</div>

[![bioRxiv](https://img.shields.io/badge/BioRxiv-10.1101/2023.11.16.567363-blue.svg)](https://www.biorxiv.org/content/10.1101/2023.11.16.567363v1)

### Abstract
----------

<p align="justify">Understanding the genomic heterogeneity of tumors is an important task in computational oncology, especially in the context of finding personalized treatments based on the genetic profile of each patient’s tumor. Tumor clustering that takes into account the temporal order of genetic events, as represented by tumor mutation trees, is a powerful approach for grouping together patients with genetically and evolutionarily similar tumors and can provide insights into discovering tumor sub-types, for more accurate clinical diagnosis and prognosis.</p>

<p align="justify">We propose <b>oncotree2vec</b>, a method for clustering tumor mutation trees by learning vector representations of mutation trees that capture the different relationships between subclones in an unsupervised manner. Learning low-dimensional tree embeddings facilitates the visualization of relations between trees in large cohorts and can be used for downstream analyses, such as deep learning approaches for single-cell multi-omics data integration.</p>

<div align="center">
<img src="./docs/fig1.png" width="70%" height="auto">
</div>

### Requirements
----------

The codebase is implemented in Python 3.11.6, with the following dependencies:
```
anytree            2.8.0
numpy              1.23.5  
pandas             2.1.3
tensorflow         2.12.0
matplotlib         3.8.2
seaborn            0.13.0
plotly             5.18.0
umap-learn         0.5.5
networkx           3.2.1
gensim             4.3.2
kaleido-core       0.1.0
```

### Arguments
----------
```
$ python oncotree2vec.py -h

usage: graph2vec [-h] -c CORPUS [-o OUTPUT_DIR] [-b BATCH_SIZE] [-e EPOCHS] [-d EMBEDDING_SIZE] [-neg NUM_NEGSAMPLE] [-lr LEARNING_RATE]
                 [--wlk_sizes [WLK_SIZES ...]] [-s SUFFIX] [-x0 AUGMENT_INDIVIDUAL_NODES] [-x1 AUGMENT_NEIGHBORHOODS] [-x2 AUGMENT_PAIRWISE_RELATIONS]
                 [-x3 AUGMENT_DIRECT_EDGES] [-x4 AUGMENT_MUTUALLY_EXCLUSIVE_RELATIONS] [-x5 AUGMENT_TREE_STRUCTURE] [-x6 AUGMENT_ROOT_CHILD_RELATIONS]
                 [-rlabel ROOT_LABEL] [-ilabel IGNORE_LABEL] [--remove_unique_words]

options:
  -h, --help            show this help message and exit
  -c CORPUS, --corpus CORPUS
                        Path to directory containing graph files to be used for clustering
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory for storing output embeddings
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of samples per training batch
  -e EPOCHS, --epochs EPOCHS
                        Number of iterations the whole dataset of graphs is traversed
  -d EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        Intended graph embedding size to be learnt
  -neg NUM_NEGSAMPLE, --num_negsample NUM_NEGSAMPLE
                        Number of negative samples to be used for training
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate to optimize the loss function
  --wlk_sizes [WLK_SIZES ...]
                        Seizes of WL kernel (i.e., degree of rooted subgraph features to be considered for representation learning)
  -s SUFFIX, --suffix SUFFIX
                        Suffix to be added to the output filenames.
  -x0 AUGMENT_INDIVIDUAL_NODES, --augment_individual_nodes AUGMENT_INDIVIDUAL_NODES
                        Number of times to augment the vocabulary for the individual nodes.
  -x1 AUGMENT_NEIGHBORHOODS, --augment_neighborhoods AUGMENT_NEIGHBORHOODS
                        Number of times to augment the vocabulary for the tree neighborhoods.
  -x2 AUGMENT_PAIRWISE_RELATIONS, --augment_pairwise_relations AUGMENT_PAIRWISE_RELATIONS
                        Number of times to augment the vocabulary for the non-adjacent pairwise relations.
  -x3 AUGMENT_DIRECT_EDGES, --augment_direct_edges AUGMENT_DIRECT_EDGES
                        Number of times to augment the vocabulary for the direct edges.
  -x4 AUGMENT_MUTUALLY_EXCLUSIVE_RELATIONS, --augment_mutually_exclusive_relations AUGMENT_MUTUALLY_EXCLUSIVE_RELATIONS
                        Number of times to augment the vocabulary for the mutually exclusive relations.
  -x5 AUGMENT_TREE_STRUCTURE, --augment_tree_structure AUGMENT_TREE_STRUCTURE
                        Number of times to augment the vocabulary for the tree structure.
  -x6 AUGMENT_ROOT_CHILD_RELATIONS, --augment_root_child_relations AUGMENT_ROOT_CHILD_RELATIONS
                        Number of times to augment the vocabulary for the root-child node relations.
  -rlabel ROOT_LABEL, --root_label ROOT_LABEL
                        Label of the neutral node (used for discarding certain node relations).
  -ilabel IGNORE_LABEL, --ignore_label IGNORE_LABEL
                        Label to be ignored when matching individual nodes or pairwise relations. (usecase: ignoring neutral clones).
  --remove_unique_words
```

### Usage example
----------
```
$ python oncotree2vec.py --corpus ../data/synthetic_data/oncotree2vec_1691495959_matching-vocabulary-sizes --embedding_size 64 --wlk_sizes 1 --augment_tree_structure 0 --augment_neighborhoods 0 --augment_individual_nodes 1 --augment_root_child_relations 0 --augment_direct_edges 0 --augment_pairwise_relations 0 --augment_mutually_exclusive_relations 0 --epochs 1000
```
