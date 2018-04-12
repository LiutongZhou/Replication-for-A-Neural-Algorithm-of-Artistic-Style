# Styletransfer
A Tensorflow implementation for A Neural Algorithm of Artistic Style (Gatys et al. 2015)

The styletransfer package is a handy implementation of the Neural Style Transfer Algorithm. It is well stuctured and documented. To get an overview of how to use the package, see [Demo.ipynb](./Demo.ipynb)

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

### What to expect next
1. The project website is under construction, you can expect to see it come out in May. 2018.
2. After 1 is done, the package will be optimized, you can expect a revision in Jan. 2018 and the release of version 1.0.
