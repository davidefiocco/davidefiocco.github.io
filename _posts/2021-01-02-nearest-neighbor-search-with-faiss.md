---
title: First steps with Faiss for k-nearest-neighbor search in large search spaces
excerpt: Nearest-neighbor search in vector spaces in very useful in a variety of tasks. How to tackle this when dealing with A LOT of vectors not fitting in RAM?
classes: wide
---

*tl;dr: The `faiss` library allows to perform nearest-neighbor search in an efficient way, scaling to several million dense vectors. Go straight to the [example code](https://github.com/davidefiocco/faiss-on-disk-example)!*

## Vector embeddings and search

A common procedure used in information retrieval and machine learning is to represent entities with low-dimensional _dense_ vectors, also known as _embeddings_. These vectors typically have a number of dimensions typically between 25 and 1000 (we call them _dense_ because the utmost majority of their components are non-zero, so they are not _sparse_).

Researchers have devised ways to compute vector embeddings for different kinds of entities: nowadays embeddings can be constructed for [words](http://www.youtube.com/watch?v=8rXD5-xhemo&t=34m35s), entire text documents, entire images, [local features](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)) in images, [nodes in graphs](https://arxiv.org/abs/1607.00653), and more (as testified by the existence of very [many "2vec" models](https://github.com/MaxwellRebo/awesome-2vec)).

Such vector representations have interesting properties (think [word2vec](http://www.youtube.com/watch?v=kEMJRjEdNzM&t=1m9s)'s). One basic property is that similar vectors (i.e. vectors close to each other according to a given _similarity metric_, like the cosine similarity or the Euclidean distance) represent entities that are somehow closely related to each other.
This feature is important as it unlocks the possibility of _searching_ for entities relevant for a given query entity.

For example, let's consider words. If we want to look up words that are related to a given word (e.g. synonyms), we can

0. Compute appropriate vector representations $S$ for all words in a vocabulary;
1. Look up the vector representation `xq` of the query word;
2. Select the vectors that are closest to `xq` (according to some given metric), among all those at step 0;
3. Retrieve all words represented by the vectors selected in step 2 (those are the final results).

One could carry out a similar procedure using documents ([Elasticsearch now supports this](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html#vector-functions) to retrieve similar documents), images ([search engines for images](https://ai.googleblog.com/2019/07/building-smily-human-centric-similar.html) can work this way) or small details in images (as done in [feature matching](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)).

If we can compute a "good" vector representation for entities (step 0 above) and have a viable algorithm to identify vectors close to a query vector (step 2, also known as _k-nearest neighbors_ search, or _k-NN search_) we can build an efficient search engine. 

In what follows we assume that step 0 has been solved for us and concentrate on how to carry out step 2, using Python and focusing on the cases of large $N$.
Solving k-NN search has great industrial relevance: companies such as [Spotify](https://github.com/spotify/annoy) and [Facebook](https://github.com/facebookresearch/faiss) have been developing libraries to solve this problem efficiently.

## k-nearest neighbors search in Python

Given a set $S$ of $d$-dimensional $N$ vectors `xb` (the _search space_) and a query vector `xq`, how can we find its nearest neighbors in $S$ using Python?  
If $N$ is large, the computation can be expensive, so it's beneficial to leverage some level of optimization offered by dedicated numerical libraries.  

In what follows we'll analyze a solution using `numpy`, `scikit-learn` and finally `faiss`, that can search among several millions of dense vectors. We will use the Euclidean distance as similarity metric for vectors (code could be modified to use other metrics).

### Linear search using `numpy`

One simple strategy is to compute the distance from `xq` to _every other vector_ in $S$, and identify the $k$ vectors that have minimum distance. As all the possible matches are evaluated, this is also called _brute force_ search.
This operation requires the computation of $N$ distances and then finding the bottom $k$ values.

Once we have all the components of the vectors in $S$ stored in a `numpy` array `xb`, we can compute the indices of the k-nearest neighbors with

```python
import numpy as np

N = 1000000
d = 100
k = 5

# read in a file of d-dimensional vectors (can check this with `assert xb.shape[1] == d`)
xb = np.random.random((N, d)).astype('float32')

# create a random d-dimensional query vector
xq = np.random.random(d)
# compute distances
distances = np.linalg.norm(xb - xq, axis = 1)
# select indices of vectors having the lowest distances from the query vector (sorted!)
neighbors = np.argpartition(distances, range(0, k))[:k]
```

This is fairly straightforward, but it's interesting to note that we're using [`numpy.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) to select the indices of the vectors having the lowest distances (`numpy.argpartition` does the job efficiently on a CPU as it uses the _introselect_ algorithm). Mind that we're calling the function using `range(0, k)` as argument, because otherwise the neighbor indices [wouldn't be sorted](https://stackoverflow.com/a/42185645/4240413) by distance from `xq`!

This strategy is simple (it takes 2 lines of code!) but it requires the entire matrix of vectors to be stored in memory. This means that if the space taken in memory by the `xb` is above the RAM available, the code won't run!

### Search with `scikit-learn`

`scikit-learn` provides a dedicated class to solve the problem, [`sklearn.neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Searching in this case would look like

```python
from sklearn.neighbors import NearestNeighbors

# set desired number of neighbors
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(xb)
# select indices of k nearest neighbors of the vectors in the input list
neighbors = neigh.kneighbors([xq], return_distance = False)
```

Using `sklearn` offers some advantages, as it automatically employs heuristics to determine if it should resort to computational tricks (like k-d trees) to reduce the number of distance calculations. A variety of metrics can be chosen to pick neighbors, and searching using multiple query vectors can be done by adding more vectors in the list passed to `neigh.kneighbors()`. 

One can also save to disk `neigh` for further reuse with
```python
from joblib import dump, load
dump(neigh, "my_fitted_nn_estimator")
```
and load it again with
```python
neigh = load("my_fitted_nn_estimator")
```

Last but not least, the `sklearn`-based code is arguably more readable. Also, the use of a dedicated library can help avoiding bugs (see e.g. the `numpy.argpartition` caveat above) that may be inadvertently introduced in the code.

However, if the search space is large (say, several million vectors), both the time needed to compute nearest-neighbors and RAM needed to carry out the search may be large. We thus need additional tricks to solve the problem!

![haystack](https://upload.wikimedia.org/wikipedia/commons/4/42/Needle_in_haystack6.jpg "Some search problems can be hard.")

### Search with `faiss`, and scale beyond RAM constraints

One library that offers a more sophisticated bag of tricks to perform the search is [`faiss`](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). 
From their [wiki on GitHub](https://github.com/facebookresearch/faiss/wiki): "_Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, **up to ones that possibly do not fit in RAM**_". 
The possibility of searching in vectors set not fitting in RAM is useful. `numpy`, `sklearn` and even Spotify's [`annoy`](https://github.com/spotify/annoy/issues/451) can't do that AFAIK.

Exact brute-force searches like the one done above with `numpy` can be replicated with the syntax:

```python
import faiss

index = faiss.index_factory(d, "Flat")
index.train(xb)
index.add(xb)
distances, neighbors = index.search(xq.reshape(1,-1).astype(np.float32), k)
```

What's cool about `faiss` is that it allows to strike a balance between accuracy (i.e. not returning all the true k-nearest neighbors, but just "good guesses"), speed and RAM requirements when dealing with large $N$. This is possible thanks to the precomputation of data structures (the _indexes_) that store vectors in a clever way, and by tweaking their parameters. The library is also designed to take advantage of GPU architectures to speed up search.

One class of tricks used to speed up search is the _pruning_ of $S$, i.e. dividing up $S$ into "buckets" (Voronoi cells in $d$ dimensions) and probing for nearest neighbors only some number `nprobe` of such buckets. While this procedure can _miss_ some of the true nearest neighbors, it can greatly accelerate the search.

`faiss` also implements _compression_ strategies to speed up the distance computation and reduce memory use. By applying methods like _product quantization_ (PQ) it is possible to obtain distances in an approximate (but faster) way, using table lookups instead of direct computation.

### A more concrete case: searching in a 1M dataset with `faiss`

The choice of which "tricks" to use for a specific problem depends on [considerations](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) about the raw input (dataset size, the spatial organization of the vectors), the available hardware (CPU vs GPU, available RAM, single node vs distributed processing), desired accuracy, speed, and total number of searches we need to perform. 
The `faiss` [wiki](https://github.com/facebookresearch/faiss/wiki) on GitHub can help evaluate the different options.

Let's examine more in detail a case in which:

- $N \approx 10^6$;
- search is performed in a Docker container running on CPU (single machine) and very few GBs of RAM are available. We can instead rely on a machine with more RAM to build the index;
- accuracy is more important than speed: ideally we'd like to have exact results;
- we plan to perform several searches (>10000) in the lifetime of an index.

To play with a realistic dataset, let's use the [GIST 1M](http://corpus-texmex.irisa.fr/) vector dataset ([GIST vectors](http://people.csail.mit.edu/torralba/code/spatialenvelope/) can be used in computer vision to represent entire images). We can download and inflate the dataset with a Linux shell using `wget` and `tar`:

```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz
```

Moving to a Python shell, with appropriate [helper functions](https://github.com/davidefiocco/faiss-on-disk-example/blob/master/src/utils.py), one can read the file `gist_base.fvecs` (3.57 GB in size) into a `numpy` array `xb` of shape `(1000000, 960)`:

```python
xb = fvecs_read("./gist/gist_base.fvecs")
```

As we plan to perform several searches (see above), we can assume that precomputing can be helpful. As we are limited by RAM but the dataset is not huge, we use pruning but not compression. We thus opt for a `IVF4000,Flat` index that [organizes the vectors in 4000 buckets in $d=960$ dimensional space](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-somewhat-then-flat):

```python
index = faiss.index_factory(xb.shape[1], f"IVF4000,Flat")
```

We then need to train the index so to cluster the vectors that are added to it.
Instead of doing this as above (i.e. with `index.train(xb)` and `index.add(xb)`), we train the index with a subset of the vectors `xb[0:batch_size]`, add vectors to sub-blocks of the index in batches of 100000 vectors each, as this allows to limit RAM consumption at search time (see also [demo_ondisk_ivf.py](https://github.com/facebookresearch/faiss/blob/master/demos/demo_ondisk_ivf.py) in the `faiss` GitHub repository):

```python
batch_size = 100000

index.train(xb[0:batch_size])
faiss.write_index(index, "trained.index")

n_batches = xb.shape[0] // batch_size
for i in range(n_batches):
    index = faiss.read_index("trained.index")
    index.add_with_ids(xb[i * batch_size:(i + 1) * batch_size], np.arange(i * batch_size, (i + 1) * batch_size))
    faiss.write_index(index, f"block_{i}.index")
```

Finally, we can save the final index to disk with:

```python
index = faiss.read_index("trained.index")
block_fnames = [f"block_{b}.index" for b in range(n_batches)]

merge_ondisk(index, block_fnames, "merged_index.ivfdata")
faiss.write_index(index, "populated.index")
```

While the above construction is a bit cumbersome and slow, it can be repaid off if we perform enough searches later down the line.  

To re-read the index from disk we can use

```python
index = faiss.read_index("populated.index")
```

May we need to recover the i-th vector in `xb`, we could use the syntax

```python
i = 42
index.make_direct_map()
index.reconstruct(i).reshape(1,-1).astype(np.float32)
```

Finally, we can perform the search for a set of 1000 query vectors `xq`. We carry out the search for a limited number of `nprobe` cells with

```python
xq = fvecs_read("./gist/gist_query.fvecs")

index.nprobe = 80
distances, neighbors = index.search(xq, k)
```

The code above retrieves the correct result for the 1st nearest neighbor in 95% of the cases (better accuracy can be obtained by setting higher values of `nprobe`). 
Memory consumption can be kept at bay: the search succeeds within a Docker container with RAM capped at 2GB, as shown by running the code with `mprof run`: 

![faiss-run](/images/2021-01-02-faiss-run.png "RAM usage over time for 1k searches on 1M GIST vectors with faiss running in a container whose RAM was capped at 2GB maximum.")

The equivalent `numpy` code needs more than 8GB to run (it crashes otherwise!), and runs slower:

![numpy-run](/images/2021-01-02-numpy-run.png "RAM usage over time for 1k searches on 1M GIST vectors with numpy")

So indeed `faiss` can be helpful to tackle cases in which `numpy` or `sklearn` would struggle.

That's it! The GIST example above can be reproduced using code on <https://github.com/davidefiocco/faiss-on-disk-example>.

### Search on

The `faiss` documentation is on its GitHub [wiki](https://github.com/facebookresearch/faiss/wiki) (the wiki contains also references to [research work](https://github.com/facebookresearch/faiss/wiki#research-foundations-of-faiss) at the foundations of the library).  
An [introductory talk](https://www.youtube.com/watch?v=Un1Q92lfhPM) about `faiss` by its core devs can be found on YouTube, and a high-level intro is also in a [FB engineering blogpost](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/).  
More code examples are available on the [`faiss` GitHub repository](https://github.com/facebookresearch/faiss/tree/master/tutorial/python).  
The website [ann-benchmarks.com](http://ann-benchmarks.com) contains the results of benchmarks run with different libraries for approximate nearest neighbors search (including `faiss`).  
Another helpful GitHub repository containing `faiss` usage tips is <https://github.com/matsui528/faiss_tips>.
