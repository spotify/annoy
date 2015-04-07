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
#include <map>
#include <queue>
#include <limits>
#include <boost/version.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/bernoulli_distribution.hpp>

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

using std::vector;
using std::string;
using std::pair;
using std::numeric_limits;
using std::make_pair;
using boost::variate_generator;
using boost::uniform_01;

#if BOOST_VERSION > 1004400
#define BOOST_RANDOM boost::random
#else
#define BOOST_RANDOM boost
#endif

template<typename T>
struct Randomness {
  // Just a dummy class to avoid code repetition.
  // Owned by the AnnoyIndex, passed around to the distance metrics

  BOOST_RANDOM::mt19937 _rng;
  BOOST_RANDOM::normal_distribution<T> _nd;
  variate_generator<BOOST_RANDOM::mt19937&, 
                    BOOST_RANDOM::normal_distribution<T> > _var_nor;
  uniform_01<T> _ud;
  variate_generator<BOOST_RANDOM::mt19937&, 
                    uniform_01<T> > _var_uni;
  BOOST_RANDOM::bernoulli_distribution<T> _bd;
  variate_generator<BOOST_RANDOM::mt19937&, 
                    BOOST_RANDOM::bernoulli_distribution<T> > _var_ber;

  Randomness() : _rng(), _nd(), _var_nor(_rng, _nd), _ud(), _var_uni(_rng, _ud), _bd(), _var_ber(_rng, _bd) {}
  inline T gaussian() {
    return _var_nor();
  }
  inline int flip() {
    return _var_ber();
  }
  inline T uniform(T min, T max) {
    return _var_uni() * (max - min) + min;
  }
};

template<typename T>
inline void normalize(T* v, int f) {
  T sq_norm = 0;
  for (int z = 0; z < f; z++)
    sq_norm += v[z] * v[z];
  T norm = sqrt(sq_norm);
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
    S label;
    S parent;
    S children[2]; // Will possibly store more than 2
    T v[1]; // We let this one overflow intentionally. Need to allocate at least 1 to make GCC happy
  };
  static inline T distance(const T* x, const T* y, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = 0, qq = 0, pq = 0;
    for (int z = 0; z < f; z++) {
      pp += x[z] * x[z];
      qq += y[z] * y[z];
      pq += x[z] * y[z];
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
    for (int z = 0; z < f; z++)
      n->v[z] = random->gaussian();
    normalize(n->v, f);
  }
};

template<typename S, typename T>
struct Euclidean {
  struct __attribute__((__packed__)) node {
    S n_descendants;
    S label;
    S parent;
    T a; // need an extra constant term to determine the offset of the plane
    S children[2];
    T v[1];
  };
  static inline T distance(const T* x, const T* y, int f) {
    T d = 0.0;
    for (int i = 0; i < f; i++) 
      d += (x[i] - y[i]) * (x[i] - y[i]);
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
    // See http://en.wikipedia.org/wiki/Bertrand_paradox_(probability)
    // We want to sample a random hyperplane out of all hyperplanes that cut the convex hull.
    // The probability of each angle is in proportion to the extent of the projection.
    // This is good because it means we try to split in the longest direction.
    // Doing this using Metropolis-Hastings sampling using 10 steps
    T* v = (T*)malloc(sizeof(T) * f); // TODO: would be really nice to get rid of this allocation
    double max_proj = 0.0;
    for (int step = 0; step < 10; step++) {
      for (int z = 0; z < f; z++)
        v[z] = random->gaussian();
      normalize(v, f);
      // Project the nodes onto the vector and calculate max and min
      T min = INFINITY, max = -INFINITY;
      for (size_t i = 0; i < nodes.size(); i++) {
        T dot = 0;
        for (int z = 0; z < f; z++)
          dot += nodes[i]->v[z] * v[z];
        if (dot > max)
          max = dot;
        if (dot < min)
          min = dot;
      }
      if (max - min > random->uniform(0, max_proj)) {
        max_proj = max - min;
        memcpy(n->v, v, sizeof(T) * f);
        n->a = -random->uniform(min, max); // Take a random split along this axis
      }
    }
    free(v);
  }
};

template<typename S, typename T, typename Distance>
class AnnoyIndex {
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
  bool _appended;
  std::map<S, S> _group_id_map;
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
    _appended = false;

    _K = (sizeof(T) * f + sizeof(S) * 2) / sizeof(S);
  }
  ~AnnoyIndex() {
    if (_loaded) {
      unload();
    } else if(_nodes) {
      free(_nodes);
    }
  }
  //update the label of an item
  S update_label(S item_id, S label) {
    if (item_id >= _n_items || item_id < 0) {
      return -1;
    }
    typename Distance::node* m = _get(item_id);  
    if (m != NULL) {
       m->label = label;
    }
    return 1;
  }

  //add one item to existing built index, return the item id 
  S add_item_to_index(const T* w, S label) {
    S item = _n_items;
    _n_items += 1; 
    _allocate_size(_n_nodes + 1);
    
    //first, move the non-leaf node stored at (item) position to the end of array; 
    memcpy(_get(_n_nodes), _get(item), _s);
    typename Distance::node* nd = _get(_n_nodes);
    if (nd->parent != 0) {
      typename Distance::node* nparent = _get(nd->parent);
      if (nparent->children[0] == item) {
         nparent->children[0] = _n_nodes;
      } else if (nparent->children[1] == item) {
         nparent->children[1] = _n_nodes;
      }
    }
    if (nd->n_descendants > _K) {
      S c0 = nd->children[0];
      typename Distance::node* nc0 = _get(c0);
      nc0->parent = _n_nodes;

      S c1 = nd->children[1];
      typename Distance::node* nc1 = _get(c1);
      nc1->parent = _n_nodes;
    }

    // if it is root, move root, 
    for (size_t r = 0; r < _roots.size(); r ++ ) {
      if (_roots[r] == item) {
          _roots[r] = _n_nodes;
      }
    }
    _n_nodes += 1;

    //store it
    add_item(item, w, label); 
    
    //put it in the trees
    for (size_t r = 0; r < _roots.size(); r ++ ) {
      S new_root = _add_item_to_tree(item, _roots[r]);      
      _roots[r] = new_root;
    }
    return item;
  }

  void add_item(S item, const T* w, S label) {
    _allocate_size(item + 1);
    typename Distance::node* n = _get(item);
    n->label = label;
    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = w[z];

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
    if (_verbose) showUpdate("has %d nodes\n", _n_nodes);
  }


  bool save(const string& filename) {
    FILE *f = fopen(filename.c_str(), "w");
    if (f == NULL)
      return false;
    if (! _appended) {
      _append_roots_at_tail();
    }

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
    off_t size = _nodes_size * _s;
    munmap(_nodes, size);
    reinitialize();
    if (_verbose) showUpdate("unloaded\n");
  }

  bool load_memory(const string& filename) {
    int fd = open(filename.c_str(), O_RDONLY, (mode_t)0400);
    if (fd == -1) {
      return false;
    }
    off_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    _n_nodes = (S)(size / _s);
    _allocate_size(_n_nodes);
    int fr = read(fd, _nodes, size);
    if (fr == -1) {
      printf("error in reading the file %s \n", filename.c_str());
    }
    printf("file contains %d nodes \n", _n_nodes);
    _appended = true;
    _remove_roots_at_tail();
    if (_roots.size() == 0) 
      return false;
    _loaded = true;
    _n_items = _get(_roots[0])->n_descendants;
    
    if (_verbose) showUpdate("found %lu roots with degree %d\n", _roots.size(), _n_items);
    return true;
  }


  bool load(const string& filename) {
    int fd = open(filename.c_str(), O_RDWR, (mode_t)0400);
    if (fd == -1)
      return false;
    off_t size = lseek(fd, 0, SEEK_END);
#ifdef MAP_POPULATE
    _nodes = (typename Distance::node*)mmap(
        0, size, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
#else
    _nodes = (typename Distance::node*)mmap(
        0, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
#endif

    _n_nodes = (S)(size / _s);
    _appended = true;
    _remove_roots_at_tail();
    if (_roots.size() == 0) 
      return false;
    _loaded = true;
    _n_items = _get(_roots[0])->n_descendants;
    
    if (_verbose) showUpdate("found %lu roots with degree %d\n", _roots.size(), _n_items);
    return true;
  }

  inline T get_distance(S i, S j) {
    const T* x = _get(i)->v;
    const T* y = _get(j)->v;
    return Distance::distance(x, y, _f);
  }

  void get_nns_by_item(S item, size_t n, vector<pair<T, S> >* result, vector<S>& label_set, size_t tn = -1) {
    const typename Distance::node* m = _get(item);
    _get_all_nns(m->v, n, result, label_set, tn);
  }


  void get_nns_by_vector(const T* w, size_t n, vector<pair<T, S> >* result, vector<S> & label_set, size_t tn = -1) {
    _get_all_nns(w, n, result, label_set, tn);
  }
 
  void get_all_groups(T dist_threshold) {
     for (size_t i = 0; i < _roots.size(); i++) {
      _get_all_groups( _roots[i], dist_threshold);
    }
  }

  void get_nns_group_by_item(S item, size_t n, vector<vector<S> >* group_results_ptr, vector<S>& label_set, size_t tn, T dist_threshold) {
    const typename Distance::node* m = _get(item);
    get_nns_group_by_vector(m->v, n, group_results_ptr, label_set, tn, dist_threshold);
  }

  void get_nns_group_by_vector(const T* v, size_t n, vector<vector<S> >* group_results_ptr, vector<S>& label_set, size_t tn, T dist_threshold) {
    vector<pair<T, S> > nns_dist;
    vector<vector<S> >& group_results = *group_results_ptr;
    _get_all_nns( v, n, &nns_dist, label_set, tn); 
    
    for (size_t i = 0; i < nns_dist.size(); i ++ ) {
     //insert into groups 
     S item = nns_dist[i].second;
     T* vw = _get(item)->v;
     bool found = false;
     for (size_t j = 0; j < group_results.size(); j ++ ) {
        for (size_t k = 0; k < group_results[j].size(); k ++ ) {
           T distance = Distance::distance(vw, _get(group_results[j][k])->v, _f);
           if (distance < dist_threshold) {
             group_results[j].push_back(item);
             found = true; 
             break;
           }
        }
        if (found) break;
     } 
     //a new group
     if (! found) {
       vector<S> g;
       g.push_back(item); 
       group_results.push_back(g);
     }
    }
    return ; 
  }
  S set_item_size(S n) {
      _allocate_size(n);
    return n;
  }

  S get_n_items() {
    return _n_items;
  }
  void verbose(bool v) {
    _verbose = v;
  }

protected:
  void _append_roots_at_tail() {
    if (_appended) {
       return ;
    }
    _allocate_size(_n_nodes + (S)_roots.size());
    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);

    // a root's n_descendants would point to the actual node
    S total_size = _n_nodes + _roots.size();
    for (size_t i = 0; i < _roots.size(); i++) {
      typename Distance::node* nd = _get(_n_nodes + (S) i );
      nd->n_descendants = _roots[i] + total_size; 
    }
    _n_nodes += _roots.size();
    _appended = true;
  }

  void _remove_roots_at_tail() {
    if (!_appended) {
       return ;
    }
    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k1 = _get(i)->n_descendants; // k1 would be the original root + total_size, rather than the duplicated root at the end of the array 
      if (k1 >=  _n_nodes) {
         _roots.push_back(k1 - _n_nodes);
      } else {
        break;
      }
    }
    _nodes_size = _n_nodes;
    _n_nodes -= (S)_roots.size(); //reclaim the space used by duplicated roots
    _appended = false;
  }
  
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

  S _make_tree(const vector<S >& indices, S item = -1) {
    if (indices.size() == 1)
      return indices[0];
    if (item == -1) {
      _allocate_size(_n_nodes + 1);
      item = _n_nodes++;
    }
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
    _get(children_0)->parent = item;
    _get(children_1)->parent = item;

    // We need to fetch m again because it might have been reallocated
    m = _get(item);
    m->children[0] = children_0;
    m->children[1] = children_1;

    return item;
  }

  S _add_item_to_tree(S item, S root) {
     typename Distance::node* nr = _get(root);
     typename Distance::node* ni = _get(item);
     if (nr->n_descendants == 1) {
       vector<S> children;
       children.push_back(item);
       children.push_back(root);
       S new_node = _make_tree(children);   
       return new_node;
     } else if (nr->n_descendants < _K - 1) {
       nr->children[nr->n_descendants] = item;
       nr->n_descendants += 1; 
       return root;
     } else if (nr->n_descendants == _K) {
       vector<S> children;
       for (size_t kk = 0; kk < nr->n_descendants; kk ++ ) {
          children.push_back(nr->children[kk]);
       }
       children.push_back(item);
       nr->n_descendants += 1; 
       _make_tree(children, root);
       return root;
     } else  { //non-leaf node
         nr = _get(root); 
         ni = _get(item);
         
         bool side = Distance::side(nr, ni->v, _f, &_random);
         S child = nr->children[side];
         S new_child = _add_item_to_tree(item, child);
         nr->children[side] = new_child;
         nr->n_descendants += 1;
         return root;
     }
          
  }
  //ggg
  void _get_all_groups(S root, T dist_threshold) {
    const typename Distance::node* nd = _get(root);
    if (nd->n_descendants == 1) { 
         return;
    } else if (nd->n_descendants <= _K) {
       for (size_t x = 0; x < nd->n_descendants; x ++ ) {
          S x_idx = nd->children[x];
          T* v_x = _get(x_idx)->v;
          for (size_t y = 0; y < x ; y ++ ) {
             S y_idx = nd->children[y];
             T* v_y = _get(y_idx)->v;
             T distance = Distance::distance(v_x, v_y, _f);
             if (distance < dist_threshold) { 
               printf("%d\t%d\t%3.3f\n", x_idx, y_idx, distance);
             }
          }
       }
    } else { 
       _get_all_groups(nd->children[1], dist_threshold);
       _get_all_groups(nd->children[0], dist_threshold);
    }
  }

 
  void _get_all_nns(const T* v, size_t n, vector<pair<T, S> >* result, vector<S>& label_set, size_t tn) {
    std::priority_queue<pair<T, S> > q;
    std::map<S, bool> r; // retrieved items map
    std::map<S, bool> cset; // category set map
    size_t c = 0;  //retrieved count
    for(size_t i = 0; i < label_set.size(); i ++) {
      cset.insert(make_pair(label_set[i], true));
    }
    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(numeric_limits<T>::infinity(), _roots[i]));
    }
    if (tn == -1) {
      tn = n * _roots.size();
    }

    vector<pair<T, S> > nns_dist;
    while (nns_dist.size() < tn && !q.empty()) {
      const pair<T, S>& top = q.top();
      S i = top.second;
      const typename Distance::node* nd = _get(top.second);
      q.pop();
      if (nd->n_descendants == 1) {
        if (r.find(i) == r.end())
        {
            if (cset.size() == 0 || cset.find(nd->label) != cset.end()) {
               nns_dist.push_back(make_pair(Distance::distance(v, _get(i)->v, _f), i));
               r.insert(make_pair(i, true));
            }
        }
      } else if (nd->n_descendants <= _K) {
	const S* dst = nd->children;
        for (size_t kk = 0; kk < nd->n_descendants; kk ++ ) {
          int w = nd->children[kk];
          if (r.find(w) == r.end()) {
            if (cset.size() == 0 || cset.find(_get(w)->label) != cset.end()) {
              nns_dist.push_back(make_pair(Distance::distance(v, _get(w)->v, _f), w));
              r.insert(make_pair(w, true));
            }
          }
        }
      } else {
        T margin = Distance::margin(nd, v, _f);
        q.push(make_pair(+margin, nd->children[1]));
        q.push(make_pair(-margin, nd->children[0]));
      }
    }

    sort(nns_dist.begin(), nns_dist.end());
    for (size_t i = 0; i < nns_dist.size() && result->size() < n; i++) {
      result->push_back(nns_dist[i]);
    }
  }
};

#endif
