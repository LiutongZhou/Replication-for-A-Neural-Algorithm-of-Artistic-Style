---
layout: single
title:  "Styletransfer Usage Demonstration"
author_profile: true
permalink: /Demo/
---

# Styletransfer
A Tensorflow implementation for A Neural Algorithm of Artistic Style (Gatys et al. 2015)

### Folder Structure
```bash
./
.
├── data
│   ├── content/ # your content photgraphs
│   └── style/ # your images of artworks
├── Demo.ipynb
├── README.md
├── report
├── styletransfer
│   ├── artist.py
│   ├── __init__.py
│   ├── io.py
│   ├── layers.py
│   ├── optimizer.py
│   └── vgg19.py
```

### Requirements
* Ubuntu 16.04
* Tensorflow-gpu 1.3.0 +
* Scipy 
* Numpy
* Pillow
* Jupyter
* Matplotlib

To quickly build the same environment as the author's, use the provided environment.yml file
```bash
conda env create -f environment.yml

```

### About
The styletransfer package is a handy implementation of the Neural Style Transfer Algorithm. It is well stuctured and documented. To get an overview of how to use the package, see [Demo.ipynb](./Demo.ipynb)
