<div align="left">
<img src="./docs/logo.png" width="300" height="auto">
</div>

[![bioRxiv](https://img.shields.io/badge/BioRxiv-10.1101/2023.11.16.567363-blue.svg)](https://www.biorxiv.org/content/10.1101/2023.11.16.567363v1)

Understanding the genomic heterogeneity of tumors is an important task in computational oncology, especially in the context of finding personalized treatments based on the genetic profile of each patient’s tumor. Tumor clustering that takes into account the temporal order of genetic events, as represented by tumor mutation trees, is a powerful approach for grouping together patients with genetically and evolutionarily similar tumors and can provide insights into discovering tumor sub-types, for more accurate clinical diagnosis and prognosis. 

We propose **oncotree2vec**, a method for clustering tumor mutation trees by learning vector representations of mutation trees that capture the different relationships between subclones in an unsupervised manner. Learning low-dimensional tree embeddings facilitates the visualization of relations between trees in large cohorts and can be used for downstream analyses, such as deep learning approaches for single-cell multi-omics data integration.


