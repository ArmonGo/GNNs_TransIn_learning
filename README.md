# A Critique on Transductive Evaluation for GNN Node Classification

This repository contains the implementation for the models included in the experimental comparison as presented in:

A Critique on Transductive Evaluation for GNN Node Classification

Abstract: Transductive learning and inductive learning are two standard paradigms widely used in deep learning for graph data. Transductive learning focuses on inference on seen data by incorporating features of the complete dataset into the training phase, while inductive learning infers on unseen instances and has the capability of generalization. The differences in learning goals require distinct data partitioning approaches, where the data is typically divided into training and test sets. Unlike tabular data, the division of graph data is constrained by its internal topology, making it difficult to obtain a clear-cut test set when considering only a single graph. As a result, most research adopts a transductive setting for graph learning and applies a masking strategy during training, which masks labels but allows the use of graph attributes, including those related to the masked entities. We argue that transductive learning and evaluation in Graph Neural Networks (GNNs) tasks, particularly for node classification, suppresses the potential of GNNs to generalize. As an alternative, we propose a data splitting approach for inductive learning to train and evaluate GNN performance for graph datasets consisting of a single graph. We demonstrate the feasibility of inductive evaluation for both training and inference on unseen data within a complete graph by means of an experiment on well-known benchmark graph datasets.

Keywords: Inductive learning; Transductive learning; GNNs; Generalization


We use the public datasets provided by the Python library "PyTorch Geometric", including Twitch (TU, PT, ES), PolBlogs, Planetoid (PubMed, Cora, CiteSeer), HeterophilousGraphDataset (Tolokers, Questions, Minesweeper), Github, Flickr, AttributedGraphDataset (Wiki, PPI, BlogCatalog), Amazon (Photo, Computers). A more detailed description of these datasets can be found on the ["Dataset Cheatsheet"](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html).
