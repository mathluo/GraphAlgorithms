## Graph Algorithms Package

- Python code for graph-cut based algorithms from Prof. Andrea Bertozzi's Research Group

- This package was created to have easier benchmarking and easier interface with existing machine learning packages in python. 

- It is currently maintained by Xiyang "Michael" Luo, email: mathluo [at] math [dot] ucla [dot] edu

## Contents
Implementation of the following algorithms: `Supervised MBO` , `Supervised Ginzburg-Landau`, `MBO-Modularity`, for both binary and full multi-class classification, and a few miscellaneous functions. 

## Goal
- This code aims to abstract away the implementation details of the diffuse-interface graph algorithms, and allow fast and easy testing of these algorithms through a simple interface. 
- Modularity is also a heavy concern of the design, allowing one to test different combinations of graph construction strategy, classes of algorithms, and parameters efficiently. 

## Usage
The main functionality is in the class `LaplacianClustering()`. The usual prodecure is as follow:
- 1. Build and specify the classifier, e.g `clf = LaplacianClustering(scheme_type = {MBO_fidelity},dt = 1)`
- 2. Load the data and ground_truth(if available) using `clf.load_data(data = , ground_truth = )`
- 3. Build the graph Laplacian(or its eigenvectors) e.g. `clf.build_graph(affinity = 'rbf, ...)`
- 4. Call `clf.fit_predict()`. This will perform the iterative clustering/classification scheme
- 5. The results after calling `fit_predict()` is stored in `clf.labels_`, which are numeric labels from {0...K-1}

After creating a classifier(e.g.`clf` above), you can always modify its parameters via `set_params()` and reuse the object. 
Though in some cases, modifying one attribute automatically erases dependent attributes. E.g., reloading data automatically clears the graph and ground_truth points. 

An instance of the `LaplacianClustering()` object(e.g. `clf` in the example above) contains a `data` field and a `graph` field. 
`clf.graph` is an instance of the `util.BuildGraph()` class, which contains the full functionalities for building a graph Laplacian from given data. If you wish to use only the graph Laplacian functionalites but not the classifiers, you can use the `util.BuildGraph()` class directly.

(See more on the Ipython Notebook demos in demo)




## References

- Bertozzi, Andrea L., and Arjuna Flenner. "Diffuse interface models on graphs for classification of high dimensional data." *Multiscale Modeling & Simulation* 10.3 (2012): 1090-1118. <a href="http://epubs.siam.org/doi/pdf/10.1137/11083109X" target="_blank"> link</a>

- Merkurjev, Ekaterina, Justin Sunu, and Andrea L. Bertozzi. "Graph MBO method for multiclass segmentation of hyperspectral stand-off detection video." *Image Processing (ICIP), 2014 IEEE International Conference on.* IEEE, 2014. <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7025138" target="_blank"> link</a>

- Hu, Huiyi, et al. "A method based on total variation for network modularity optimization using the MBO scheme." *SIAM Journal on Applied Mathematics* 73.6 (2013): 2224-2246. <a href="http://epubs.siam.org/doi/pdf/10.1137/130917387" target="_blank"> link</a>

- Hu, Huiyi, et al. "A method based on total variation for network modularity optimization using the MBO scheme." *SIAM Journal on Applied Mathematics* 73.6 (2013): 2224-2246. <a href="http://www.math.ucla.edu/~bertozzi/papers/EMMCVPRfinal.pdf" target="_blank"> link</a>


## Note:
- Current version requires the installation of `sklearn` module. Nearest Neighbor graphs use the `KDTree` algorithm implemented in sci-kit learn's `sklearn.neighbors.kneighbors_graph` routine. Full graphs are computed using scipy's `cdist` function.

- The unsupervised methods: `MBO_modularity` and `MBO_chan_veses` needs further adjustments and will be updated soon. 

- The tutorials are in Jupyter Notebook format(Not Ipython Notebook 2). Static webpages for the notebooks are also provided.
