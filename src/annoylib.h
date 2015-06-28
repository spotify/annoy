// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef ANNOYLIB_H
#define ANNOYLIB_H

#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>

#ifdef __MINGW32__
#include "mman.h"
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>

// This allows others to supply their own logger / error printer without
// requiring Annoy to import their headers. See RcppAnnoy for a use case.
#ifndef __ERROR_PRINTER_OVERRIDE__
  #define showUpdate(...) { fprintf(stderr, __VA_ARGS__ ); }
#else
  #define showUpdate(...) { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif

#ifndef NO_PACKED_STRUCTS
#define PACKED_STRUCTS_EXTRA __attribute__((__packed__))
// TODO: this is turned on by default, but may not work for all architectures! Need to investigate.
#endif

// This allows others to supply their own random number generator function
// which is assumed to be uniformly distributed over over 0 .. RAND_MAX
inline long int randomDraw() {
#ifndef __UNIFORM_RAND_OVERRIDE__
  return rand();
#else
  return __UNIFORM_RAND_OVERRIDE__;
#endif
}

using std::vector;
using std::string;
using std::pair;
using std::numeric_limits;
using std::make_pair;

template<typename T>
struct Randomness {
  // Just a dummy class to avoid code repetition.
  // Owned by the AnnoyIndex, passed around to the distance metrics
  inline int flip() {
    return randomDraw() % 2;
  }
  inline int index(int n) {
    return randomDraw() % n;
  }
  inline void index2(int n, int* i, int* j) {
    // Draw two random variables so that i != j
    *i = index(n);
    *j = index(n-1);
    *j += (*j >= *i);
  }
};

template<typename T>
inline T get_sq_norm(const T* v, int f) {
  T sq_norm = 0;
  for (int z = 0; z < f; z++)
    sq_norm += v[z] * v[z];
  return sq_norm;
}

template<typename T>
inline T get_norm(const T* v, int f) {
  return sqrt(get_sq_norm(v, f));
}

template<typename T>
inline void normalize(T* v, int f) {
  T norm = get_norm(v, f);
  for (int z = 0; z < f; z++)
    v[z] /= norm;
}


template<typename S, typename T>
struct Angular {
  struct PACKED_STRUCTS_EXTRA node {
    /*
     * We store a binary tree where each node has two things
     * - A vector associated with it
     * - Two children
     * All nodes occupy the same amount of memory
     * All nodes with n_descendants == 1 are leaf nodes.
     * A memory optimization is that for nodes with 2 <= n_descendants <= K,
     * we skip the vector. Instead we store a list of all descendants. K is
     * determined by the number of items that fits in the same space.
     * For nodes with n_descendants == 1 or > K, there is always a
     * corresponding vector. 
     * Note that we can't really do sizeof(node<T>) because we cheat and allocate
     * more memory to be able to fit the vector outside
     */
    S n_descendants;
    union {
      S children[2]; // Will possibly store more than 2 - only for non-leaves
      T norm; // For leaves
    };
    T v[1]; // We let this one overflow intentionally. Need to allocate at least 1 to make GCC happy
  };
  static inline void init(node* a, int f) {
    a->norm = get_norm(a->v, f);
  }
  static inline T distance(const node* a, const node* b, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = 0, qq = 0, pq = 0;
    const T* x = a->v;
    const T* y = b->v;
    for (int z = 0; z < f; z++, x++, y++) {
      pp += (*x) * (*x);
      qq += (*y) * (*y);
      pq += (*x) * (*y);
    }
    T ppqq = pp * qq;
    if (ppqq > 0) return 2.0 - 2.0 * pq / sqrt(ppqq);
    else return 2.0; // cos is 0
  }
  static inline T margin(const node* n, const T* y, int f) {
    T dot = 0;
    for (int z = 0; z < f; z++)
      dot += n->v[z] * y[z];
    return dot;
  }
  static inline bool side(const node* n, const T* y, int f, Randomness<T>* random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random->flip();
  }
  static inline void create_split(const vector<node*>& nodes, int f, Randomness<T>* random, node* n) {
    // Sample two random points from the set of nodes
    // Calculate the hyperplane equidistant from them
    int i, j;
    random->index2(nodes.size(), &i, &j);
    T i_norm = get_norm(nodes[i]->v, f);
    T j_norm = get_norm(nodes[j]->v, f);
    for (int z = 0; z < f; z++)
      n->v[z] = nodes[i]->v[z] / i_norm - nodes[j]->v[z] / j_norm;
    normalize(n->v, f);
  }
};

template<typename S, typename T>
struct Euclidean {
  struct __attribute__((__packed__)) node {
    S n_descendants;
    union {
      struct {
	T a; // need an extra constant term to determine the offset of the plane
	S children[2];
      }; // For non-leaves
      T sq_norm; // For leaves
    };
    T v[1];
  };
  static inline void init(node* a, int f) {
    a->sq_norm = get_sq_norm(a->v, f);
  }
  static inline T distance(const node* a, const node* b, int f) {
    T d = 0.0;
    const T* x = a->v;
    const T* y = b->v;
    for (int i = 0; i < f; i++, x++, y++)
      d += ((*x) - (*y)) * ((*x) - (*y));
    return d;
  }
  static inline T margin(const node* n, const T* y, int f) {
    T dot = n->a;
    for (int z = 0; z < f; z++)
      dot += n->v[z] * y[z];
    return dot;
  }
  static inline bool side(const node* n, const T* y, int f, Randomness<T>* random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random->flip();
  }
  static inline void create_split(const vector<node*>& nodes, int f, Randomness<T>* random, node* n) {
    // Same as Angular version, but no normalization and has to compute the offset too
    int i, j;
    random->index2(nodes.size(), &i, &j);
    n->a = 0.0;
    for (int z = 0; z < f; z++) {
      n->v[z] = nodes[i]->v[z] - nodes[j]->v[z];
      n->a += -n->v[z] * (nodes[i]->v[z] + nodes[j]->v[z]) / 2;
    }
  }
};

template<typename S, typename T>
class AnnoyIndexInterface {
 public:
  virtual ~AnnoyIndexInterface() {};
  virtual void add_item(S item, const T* w) = 0;
  virtual void build(int q) = 0;
  virtual bool save(const char* filename) = 0;
  virtual void reinitialize() = 0;
  virtual void unload() = 0;
  virtual bool load(const char* filename) = 0;
  virtual T get_distance(S i, S j) = 0;
  virtual void get_nns_by_item(S item, size_t n, vector<S>* result, size_t search_k=-1) = 0;
  virtual void get_nns_by_vector(const T* w, size_t n, vector<S>* result, size_t search_k=-1) = 0;
  virtual S get_n_items() = 0;
  virtual void verbose(bool v) = 0;
  virtual void get_item(S item, vector<T>* v) = 0;
};

template<typename S, typename T, typename Distance>
  class AnnoyIndex : public AnnoyIndexInterface<S, T> {
  /*
   * We use random projection to build a forest of binary trees of all items.
   * Basically just split the hyperspace into two sides by a hyperplane,
   * then recursively split each of those subtrees etc.
   * We create a tree like this q times. The default q is determined automatically
   * in such a way that we at most use 2x as much memory as the vectors take.
   */
protected:
  int _f;
  size_t _s;
  S _n_items;
  Randomness<T> _random;
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  S _n_nodes;
  S _nodes_size;
  vector<S> _roots;
  S _K;
  bool _loaded;
  bool _verbose;
public:
  AnnoyIndex(int f) : _random() {
    _f = f;
    _s = sizeof(typename Distance::node) + sizeof(T) * (f - 1); // Size of each node
    // Note that we need to subtract one because we already allocated it
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _nodes = NULL;
    _loaded = false;
    _verbose = false;

    _K = (sizeof(T) * f + sizeof(S) * 2) / sizeof(S);
  }
  ~AnnoyIndex() {
    if (_loaded) {
      unload();
    } else if(_nodes) {
      free(_nodes);
    }
  }

  void add_item(S item, const T* w) {
    _allocate_size(item + 1);
    typename Distance::node* n = _get(item);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    memcpy(n->v, w, _f * sizeof(T));
    Distance::init(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;
  }

  void build(int q) {
    _n_nodes = _n_items;
    while (1) {
      if (q == -1 && _n_nodes >= _n_items * 2)
        break;
      if (q != -1 && _roots.size() >= (size_t)q)
        break;
      if (_verbose) showUpdate("pass %zd...\n", _roots.size());

      vector<S> indices;
      for (S i = 0; i < _n_items; i++)
        indices.push_back(i);

      _roots.push_back(_make_tree(indices));
    }
    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    _allocate_size(_n_nodes + (S)_roots.size());
    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);
    _n_nodes += _roots.size();

    if (_verbose) showUpdate("has %d nodes\n", _n_nodes);
  }

  bool save(const char* filename) {
    FILE *f = fopen(filename, "w");
    if (f == NULL)
      return false;

    fwrite(_nodes, _s, _n_nodes, f);

    fclose(f);

    free(_nodes);
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _nodes = NULL;
    _roots.clear();
    return load(filename);
  }

  void reinitialize() {
    _nodes = NULL;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _roots.clear();
  }

  void unload() {
    off_t size = _n_nodes * _s;
    munmap(_nodes, size);
    reinitialize();
    if (_verbose) showUpdate("unloaded\n");
  }

  bool load(const char* filename) {
    int fd = open(filename, O_RDONLY, (mode_t)0400);
    if (fd == -1)
      return false;
    off_t size = lseek(fd, 0, SEEK_END);
#ifdef MAP_POPULATE
    _nodes = (typename Distance::node*)mmap(
        0, size, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
#else
    _nodes = (typename Distance::node*)mmap(
        0, size, PROT_READ, MAP_SHARED, fd, 0);
#endif

    _n_nodes = (S)(size / _s);

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    S m = -1;
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == -1 || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    _loaded = true;
    _n_items = m;
    if (_verbose) showUpdate("found %lu roots with degree %d\n", _roots.size(), m);
    return true;
  }

  T get_distance(S i, S j) {
    return Distance::distance(_get(i), _get(j), _f);
  }

  void get_nns_by_item(S item, size_t n, vector<S>* result, size_t search_k=-1) {
    const typename Distance::node* m = _get(item);
    _get_all_nns(m, n, result, search_k);
  }

  void get_nns_by_vector(const T* w, size_t n, vector<S>* result, size_t search_k=-1) {
    typename Distance::node* m = (typename Distance::node*)malloc(_s);
    memcpy(m->v, w, _f * sizeof(T));
    Distance::init(m, _f);
    _get_all_nns(m, n, result, search_k);
    free(m);
  }
  S get_n_items() {
    return _n_items;
  }
  void verbose(bool v) {
    _verbose = v;
  }

  void get_item(S item, vector<T>* v) {
    typename Distance::node* m = _get(item);
    for (int z = 0; z < _f; z++)
      v->push_back(m->v[z]);
  }

protected:
  void _allocate_size(S n) {
    if (n > _nodes_size) {
      S new_nodes_size = (_nodes_size + 1) * 2;
      if (n > new_nodes_size)
        new_nodes_size = n;
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset((char *)_nodes + (_nodes_size * _s)/sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
      _nodes_size = new_nodes_size;
    }
  }

  inline typename Distance::node* _get(S i) {
    return (typename Distance::node*)((char *)_nodes + (_s * i)/sizeof(char));
  }

  S _make_tree(const vector<S >& indices) {
    if (indices.size() == 1)
      return indices[0];

    _allocate_size(_n_nodes + 1);
    S item = _n_nodes++;
    typename Distance::node* m = _get(item);
    m->n_descendants = (S)indices.size();

    if (indices.size() <= (size_t)_K) {
      // Using std::copy instead of a loop seems to resolve issues #3 and #13,
      // probably because gcc 4.8 goes overboard with optimizations.
      copy(indices.begin(), indices.end(), m->children);
      return item;
    }

    vector<S> children_indices[2];
    for (int attempt = 0; attempt < 20; attempt ++) {
      /*
       * Create a random hyperplane.
       * If all points end up on the same time, we try again.
       * We could in principle *construct* a plane so that we split
       * all items evenly, but I think that could violate the guarantees
       * given by just picking a hyperplane at random
       */
      vector<typename Distance::node*> children;

      for (size_t i = 0; i < indices.size(); i++) {
        // TODO: this loop isn't needed for the angular distance, because
        // we can just split by a random vector and it's fine. For Euclidean
        // distance we need it to calculate the offset
        S j = indices[i];
        typename Distance::node* n = _get(j);
        if (n)
          children.push_back(n);
      }

      Distance::create_split(children, _f, &_random, m);

      children_indices[0].clear();
      children_indices[1].clear();

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        typename Distance::node* n = _get(j);
        if (n) {
          bool side = Distance::side(m, n->v, _f, &_random);
          children_indices[side].push_back(j);
        }
      }

      if (children_indices[0].size() > 0 && children_indices[1].size() > 0) {
        break;
      }
    }

    while (children_indices[0].size() == 0 || children_indices[1].size() == 0) {
      // If we didn't find a hyperplane, just randomize sides as a last option
      if (_verbose && indices.size() > 100000)
        showUpdate("Failed splitting %lu items\n", indices.size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      for (int z = 0; z < _f; z++)
        m->v[z] = 0.0;

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    S children_0 = _make_tree(children_indices[0]);
    S children_1 = _make_tree(children_indices[1]);

    // We need to fetch m again because it might have been reallocated
    m = _get(item);
    m->children[0] = children_0;
    m->children[1] = children_1;

    return item;
  }

  void _get_all_nns(const typename Distance::node* m, size_t n, vector<S>* result, size_t search_k) {
    std::priority_queue<pair<T, S> > q;

    if (search_k == -1)
      search_k = n * _roots.size(); // slightly arbitrary default value

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(numeric_limits<T>::infinity(), _roots[i]));
    }

    vector<S> nns;
    while (nns.size() < search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      const typename Distance::node* nd = _get(top.second);
      q.pop();
      if (nd->n_descendants == 1) {
        nns.push_back(i);
      } else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), nd->children, &dst[nd->n_descendants]);
      } else {
        T margin = Distance::margin(nd, m->v, _f);
        q.push(make_pair(std::min(d, +margin), nd->children[1]));
        q.push(make_pair(std::min(d, -margin), nd->children[0]));
      }
    }

    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    sort(nns.begin(), nns.end());
    vector<pair<T, S> > nns_dist;
    S last = -1;
    for (size_t i = 0; i < nns.size(); i++) {
      S j = nns[i];
      if (j == last)
        continue;
      last = j;
      nns_dist.push_back(make_pair(Distance::distance(m, _get(j), _f), j));
    }

    size_t o = nns_dist.size();
    size_t p = n < o ? n : o; // Return this many items
    std::partial_sort(&nns_dist[0], &nns_dist[p], &nns_dist[o]);
    for (size_t i = 0; i < p; i++) {
      result->push_back(nns_dist[i].second);
    }
  }
};

#endif
// vim: tabstop=2 shiftwidth=2
