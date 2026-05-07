# MMGCN: MiRNA-disease Association Prediction via Multi-view Graph Attention Fusion Network

This repository provides the implementation of **MMGCN**, a multi-view graph attention network for predicting potential miRNA–disease associations.

The corresponding paper is:

**MiRNA-disease Association Prediction via Multi-view Graph Attention Fusion Network**

## Overview

MicroRNAs (miRNAs) play important regulatory roles in many biological processes and human diseases. However, experimentally identifying miRNA–disease associations is costly and time-consuming.

MMGCN integrates multiple biological similarity views and uses graph attention networks to learn representations of miRNAs and diseases. The model constructs three heterogeneous graphs based on:

- miRNA sequence similarity
- miRNA functional similarity
- miRNA semantic similarity
- disease semantic similarity
- known miRNA–disease associations

The learned miRNA and disease embeddings from different views are fused for association prediction.

## Dataset

The dataset is based on **HMDD v2.0**.

It contains:

- 550 miRNAs
- 328 diseases
- 6,088 experimentally confirmed miRNA–disease associations
- three types of miRNA similarity
- one type of disease similarity

The main data files should be placed under the `data/` directory:

```text
data/
├── m_d_associations.txt
├── disease_semantic_similarity.txt
├── miRNA_functional_similarity.txt
├── miRNA_sequence_similarity.txt
└── miRNA_semantic_similarity.txt
