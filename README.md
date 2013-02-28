## What is this?

Annoy (Approximate Nearest Neighbors Something Something) is a C++ library with Python bindings to do approximate nearest neighbor search by cosine. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

You want to use this if:

* You are doing nearest neighbor search using cosine
* You don't have too many dimensions (like <100)
* You have a few million items that easily fit in RAM
* Memory usage is a concern
* You want to share memory between multiple processes
* Index creation is separate from lookup (in particular you can not add more items once the tree has been created)
* You're using Python (although you could also use it from C of course)

We use it at [Spotify](http://www.spotify.com/) for recommendations. After running matrix factorization algorithms, every user/item can be represented as a vector in f-dimensional space. This library helps us search for similar users/items.

It was all built by Erik Bernhardsson in a couple of afternoons during [Hack Week](http://labs.spotify.com/2013/02/15/organizing-a-hack-week/).

## Code example

```python

	f = 40
    t = AnnoyIndex(f)
    for i in xrange(n):
        v = []
        for z in xrange(f):
            v.append(random.gauss(0, 1))
        t.add_item(i, v)

    t.build(50) # 50 trees
    t.save('test.tree')
    
    # …
    
    u = AnnoyIndex(f)
    u.load('test.tree') # super fast, will just mmap the file
    print u.get_nns_by_item(0, 1000) # will find the 1000 nearest neighbors

'''

Right now it only accepts integers as identifiers for items. Note that it will allocate memory for max(id)+1 items because it generally assumes you will have items 0 … n.

## How does it work

Using [random projections](http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection) and by building up a tree. At every intermediate node in the tree, a random hyperplane is chosen, which divides the space into two subspaces.

We do this k times so that we get a forest of trees. k has to be tuned to your need, by looking at what tradeoff you have between precision and performance. In practice k should probably be on the order of dimensionality. Empirically, if most items have a positive cosine (which is the case of most non-negative matrix factorization algorithm), you need much fewer trees.

## Source

It's all written in horrible C++ with a handful optimizations for memory usage. You have been warned :)

## Future stuff

* Other distance functions (Euclidean, dot product, etc)
* Better support for other languages
* More performance tweaks
