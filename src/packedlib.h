// Copyright (c) 2013 Spotify AB
// Copyright (c) 2018 viper.craft@gmail.com
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

// WARNING! you need to enable SSSE3 instructions for this file! (-mssse3)

#pragma once

#include "packutils.h"
#include "annoylib.h"
#include <stdexcept>
#include <deque>
#include <memory>
#include <iostream>

template<typename S, typename T, typename Distance, typename Random>
class PackedAnnoyIndexer {
  /*
     This is AnnoyIndex class divided into indexer and searcher
     to eliminate write&read spare states.

     Also this two classes provided different nodes packing scheme
     including float[-1:+1] to uint16 pack
     Some stats: packs about to 1.8x or more of original annoy index!
     unpack speed has no penalty vs not packed!
   */
public:
  typedef Distance D;
  typedef typename D::template Node<S, T> Node;

private:
  static constexpr S const _K_mask = S(1) << S(sizeof(S) * 8 - 1);

protected:
  const int _f;
  S  _s; // Size of each node
  S const _K; // Max number of descendants to fit into node
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  vector<S> _roots;
  S _n_items;
  S _nodes_size; // capacity of the write buffer
  S _n_nodes; // number of nodes
  Random _random;
  bool _loaded;
  bool _verbose;
  int _fd;

  typedef std::vector<S> indices_list_t;
  std::deque<indices_list_t>  _indices_lists;
public:

  PackedAnnoyIndexer(int f, S idx_block_len ) 
    : _f(f)
    , _s(offsetof(Node, v) + _f * sizeof(T))
    , _K(idx_block_len)
    , _verbose(false)
  {
    reinitialize(); // Reset everything
  }

  ~PackedAnnoyIndexer() {
    unload();
  }

  int get_f() const {
    return _f;
  }

  void add_item(S item, const T* w) {
    _allocate_size(item + 1);
    Node* n = _get(item);

    D::zero_value(n);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    memcpy(n->v, w, sizeof(T) * _f);

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;
  }

  void build(int q) {
    if (_loaded) {
      // TODO: throw exception
      showUpdate("You can't build a loaded index\n");
      return;
    }

    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;
    while (1) {
      if (q == -1 && _n_nodes >= _n_items * 2)
        break;
      if (q != -1 && _roots.size() >= (size_t)q)
        break;
      if (_verbose) showUpdate("pass %zd...\n", _roots.size());

      vector<S> indices;
      for (S i = 0; i < _n_items; i++) {
          if (_get(i)->n_descendants >= 1) // Issue #223
          indices.push_back(i);
      }

      _roots.push_back(_make_tree(std::move(indices), true));
    }

    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    S const nroots = _roots.size();
    _allocate_size(_n_nodes + nroots);
    for (S i = 0; i < nroots; i++)
    {
      memcpy(_get(_n_nodes + i), _get(_roots[i]), _s);
    }
    _n_nodes += nroots;

    if (_verbose) showUpdate("has %d nodes\n", _n_nodes);
  }
  
  void unbuild() {
    if (_loaded) {
      showUpdate("You can't unbuild a loaded index\n");
      return;
    }

    _roots.clear();
    _n_nodes = _n_items;
  }

  static inline uint32_t maxbits(uint32_t const v)
  {   
    return v == 0 ? 0 : 32 - __builtin_clz(v);
  }

  static uint32_t get_max_bits( uint32_t const *b, uint32_t const *e )
  {
    uint32_t max = 0, prev = 0;
    for( ; b != e; ++b )
    {
        max |= *b - prev;
        prev = *b;
    }

    return maxbits(max);
  }

  bool save(const char* filename) {
    // calc size of packed node
    S packed_size = offsetof(Node, v) + _f * sizeof(uint16_t);
    // allocate new buffer
    void *packed_nodes = malloc(packed_size * _n_nodes);
    std::unique_ptr<void, decltype( &free )> membuf_safer(packed_nodes, free);
    // repack nodes to the new buffer directly
    for (S i = 0; i < _n_nodes; i++) {
      Node *node = get_node_ptr<S, Node>(_nodes, _s, i);
      Node *packed = get_node_ptr<S, Node>(packed_nodes, packed_size, i);
      // pack and copy to new buffer      
      memcpy(packed, node, offsetof(Node, v));
      pack_float_vector_i16(node->v, (uint16_t*)packed->v, _f);
    }

    // get indices stats?
    S const iblocks = _indices_lists.size();
    //*
    uint32_t min_max_bit = 100, max_max_bit = 0;
    size_t total_bits = 0, total_size = 0;
    for( auto const &i : _indices_lists )
    {
      total_size += i.size();
      uint32_t const *idx = i.data(), *e = idx + i.size();
      //std::sort(idx, e);
      uint32_t mb = get_max_bits(idx, e);
      min_max_bit = std::min(min_max_bit, mb);
      max_max_bit = std::max(max_max_bit, mb);
      total_bits += mb;
    }

    std::cout << "total normal=" << _n_items << " total_nodes=" << _n_nodes
          << "\nmin_max_bit=" << min_max_bit << " max_max_bit=" << max_max_bit << " avg_mb=" << total_bits / double(iblocks)
          << "\ntotal size of indices=" << iblocks * _K * sizeof(S) << " numbers of blocks=" << iblocks
          << std::endl;

    
    auto iblock_avg_sz = total_size / double(iblocks);
    std::cout << "iblock avg sz=" << iblock_avg_sz << " waste=" << 1.0 - (iblock_avg_sz / (_K - 1 )) << std::endl;
    //*/

    FILE *f = fopen(filename, "wb");
    if (f == NULL)
      return false;


    // write indices first
    S index_write_block[_K];
    S header_size = 1;
    // find first not fully filled node and use it as preudo-header ;)
    // yes we very love hacks :)
    for( auto const &i : _indices_lists )
    {
      S const isz = i.size();
      index_write_block[0] = isz;
      S *data_start = index_write_block + 1;
      memset(data_start, 0, sizeof(index_write_block) - sizeof(S));
      if( _K - isz > header_size )
      {
          // found(!)
          // fill backward
          index_write_block[_K - header_size] = iblocks;
          // prevent from writting this for ever block
          header_size = _K << 1;
      }      
      memcpy(data_start, i.data(), isz * sizeof(S));
      fwrite(index_write_block, sizeof(index_write_block), 1, f);
    }
    // and nodes data
    fwrite(packed_nodes, packed_size, _n_nodes, f);
    fclose(f);

    unload();

    return true;
    //return load(filename);
  }

  void reinitialize() {
    _fd = 0;
    _nodes = nullptr;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _roots.clear();
  }

  void unload() {
    if (_nodes) {
      // We have heap allocated data
      free(_nodes);
    }
    reinitialize();
    if (_verbose) showUpdate("unloaded\n");
  }

  void set_seed(int seed) {
    _random.set_seed(seed);
  }

  void verbose(bool v) {
    _verbose = v;
  }

protected:
  void _allocate_size(S n) {
    if (n > _nodes_size) {
      const double reallocation_factor = 1.3;
      S new_nodes_size = std::max(n,
                  (S)((_nodes_size + 1) * reallocation_factor));
      if (_verbose) showUpdate("Reallocating to %d nodes\n", new_nodes_size);
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset((char *)_nodes + (_nodes_size * _s)/sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
      _nodes_size = new_nodes_size;
    }
  }


  inline Node* _get(S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  S _append_indices( vector<S> && indices )
  {
    S i = _indices_lists.size();
    _indices_lists.emplace_back( std::move(indices) );
    return i | _K_mask;
  }

  S _make_tree(vector<S > && indices, bool is_root) {
    S const isz = indices.size();
    S const max_n_descendants = _K - 1;
    // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
    // There's some regrettable complications caused by the problem that root nodes have to be "special":
    // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
    // 2. Root nodes with only 1 child need to be a "dummy" parent
    // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
    if (isz == 1 && !is_root)
      return indices[0];

    if (isz <= max_n_descendants && (!is_root || _n_items <= max_n_descendants || isz == 1)) {
      if( is_root )
        throw std::runtime_error("cannot make root here!");
      return _append_indices(std::move(indices));
    }

    // map indices to nodes pointers
    vector<Node*> children;
    children.reserve(isz);
    for (S j : indices) {
      Node* n = _get(j);
      children.push_back(n);
    }

    vector<S> children_indices[2];
    children_indices[0].reserve(isz / 2 + 1);
    children_indices[1].reserve(isz / 2 + 1);
    Node* m = (Node*)malloc(_s); // TODO: avoid
    D::create_split(children, _f, _s, _random, m);

    for (S j : indices) {
      Node* n = _get(j);
      bool side = D::side(m, n->v, _f, _random);
      children_indices[side].push_back(j);
    }

    // If we didn't find a hyperplane, just randomize sides as a last option
    while (children_indices[0].empty() || children_indices[1].empty()) {
      if (_verbose && false)
        showUpdate("\tNo hyperplane found (left has %ld children, right has %ld children)\n",
          children_indices[0].size(), children_indices[1].size());
      if (_verbose && isz > 100000)
        showUpdate("Failed splitting %lu items\n", indices.size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      memset(m->v, 0, _f * sizeof(T));

      for (S j : indices) {
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());

    m->n_descendants = is_root ? _n_items : (S)indices.size();
    for (int side = 0; side < 2; side++) {
      // run _make_tree for the smallest child first (for cache locality)
      m->children[side^flip] = _make_tree(std::move(children_indices[side^flip]), false);
    }

    _allocate_size(_n_nodes + 1);
    S item = _n_nodes++;
    memcpy(_get(item), m, _s);
    free(m);

    return item;
  }
};


template<typename S, typename T, typename Distance>
class PackedAnnoySearcher {
    /*
     This is AnnoyIndex class divided into indexer and searcher
     to eliminate write&read spare states.

     Also this two classes provided different nodes packing scheme
     including float[-1:+1] to uint16 pack
     Some stats: packs about to 1.8x or more of original annoy index!
   */
public:
  typedef Distance D;
  typedef typename Distance::PackedFloatType PackedFloatType;
  //typedef EuclideanPacked16 SD;
  typedef typename D::template Node<S, T> Node;
private:
  static constexpr S const _K_mask = S(1) << S(sizeof(S) * 8 - 1);
  static constexpr S const _K_mask_clear = _K_mask - 1;
protected:
  int const _f;
  S const _s; // Size of each node
  S const _K; // Max number of descendants to fit into node
  S _n_items; // number of ordinal nodes(i.e leaf)
  void const *_nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  void const *_mmaped;
  std::vector<S> _roots;  
  size_t _size_of_mapping;
  int _fd;
public:

  PackedAnnoySearcher( int f, S idx_block_len ) 
    : _f(f)
    , _s(offsetof(Node, v) + _f * sizeof(PackedFloatType))
    , _K(idx_block_len)
    , _n_items(0)
    , _nodes(nullptr)
    , _mmaped(nullptr)
    , _size_of_mapping(0)
    , _fd(0)
  {
    // check size of node must be multiply of 16
    if( _s % 16 )
      throw std::runtime_error("size of the node must be multiply of 16 bytes, consider using different config!");

    if( (_K * sizeof(S)) % 16 )
      throw std::runtime_error("size of the index-node must be multiply of 16 bytes, consider using different idx_block_len!");
  }

  ~PackedAnnoySearcher()
  {
    if( _fd )
    {
      close(_fd);
      if( _mmaped != MAP_FAILED )
      {
        // INFO: no need to call munlock, it will be automatically
        munmap(const_cast<void*>(_mmaped), _size_of_mapping);
      }
    }
  }

  void get_item(S item, T* v) const {
    Node* m = _get(item);
    decode_vector_i16_f32((uint16_t const*)m->v, v, _f);
  }

  bool load(const char* filename, bool need_mlock) {
    _fd = open(filename, O_RDONLY, (int)0400);
    if (_fd == -1) {
      _fd = 0;
      return false;
    }
    off_t size = lseek(_fd, 0, SEEK_END);
    _mmaped = mmap(0, size, PROT_READ, MAP_SHARED, _fd, 0);
    if( _mmaped == MAP_FAILED )
      return false;

    // keep size only for dtor()
    _size_of_mapping = size;

    if( need_mlock )
    {
      mlock(_mmaped, size);
    }

    _size_of_mapping = size;

    // read pseudo-header
    S const *index_start = (S const*)_mmaped;
    S const size_needed = 1;
    S nblocks = 0;
    for( S const *line = index_start; /* win or die */; line += _K )
    {
      if( (_K - *line) > size_needed ) 
      {
        nblocks = line[_K - 1];
        break; // win
      }
    }

    S nindices = nblocks * _K;

    // get offset to the vector data start
    _nodes = (void*)(index_start + nindices);

    size_t sizeof_indices = nindices * sizeof(S);

    S n_nodes = (S)((size - sizeof_indices)  / _s);

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    _roots.clear();
    S m = -1;
    for (S i = n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == S(-1) || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (_roots.size() > 1 && _get(_roots.front())->children[0] == _get(_roots.back())->children[0])
      _roots.pop_back();
    _n_items = m;

    return true;
  }

  T get_distance(S i, S j) const {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    float __attribute__((aligned(16))) mv[_f];
    const Node* m = _get(item);
    decode_vector_i16_f32((uint16_t const*)m->v, mv, _f);
    //memcpy(mv, m->v, sizeof(float) * _f);
    _get_all_nns(mv, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const {
    return _n_items;
  }

private:
  inline Node* _get(S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  void _get_all_nns(const T* v, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    char node_alloc_buf[offsetof(Node, v) + _f * sizeof(T)]; // alloc on stack!
    Node* v_node = (Node *)node_alloc_buf;

    D::template zero_value<Node>(v_node);
    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    typedef std::pair<T, S> qpair_t;
    typedef std::vector<qpair_t> qvector_t;

    size_t const roots_size = _roots.size();

    if (search_k == (size_t)-1)
      search_k = n * roots_size; // slightly arbitrary default value

    // reduce prealloc(maybe user give us wrong values?)
    size_t const prealloc = std::min(search_k, roots_size * 128);

    qvector_t qvector;
    qvector.reserve(prealloc); // prealloc queue

    for ( S r : _roots ) {
      qvector.emplace_back(Distance::template pq_initial_value<T>(), r);
    }

    std::priority_queue<qpair_t, qvector_t> q( std::less<qpair_t>(), std::move(qvector) );

    std::vector<S> nns;
    nns.reserve(search_k);
    while (nns.size() < search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S fi = top.second, i = fi & _K_mask_clear;
      q.pop();
      
      if( !(fi & _K_mask) )
      {
        // ordinal node
        Node* nd = _get(i);
        if (nd->n_descendants == 1 && i < _n_items) {
          nns.push_back(i);
        } else {
          // split node
          T margin = D::margin(nd, v, _f);
          q.emplace(D::pq_distance(d, margin, 1), nd->children[1]);
          q.emplace(D::pq_distance(d, margin, 0), nd->children[0]);
        }
      }
      else
      {
        // index only node
        S const *idx = (S const*)_mmaped + i * _K;
        S const *dst = idx + 1;
        nns.insert(nns.end(), dst, dst + *idx);
      }
    }

    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    // NOTE: this only need to improve some quality, for speed is better to avoid sorting
    std::sort(nns.begin(), nns.end());
    vector<pair<T, S> > nns_dist;
    nns_dist.reserve(nns.size());
    S last = -1;
    for (S j : nns) {
      if (j == last)
        continue;
      last = j;
      Node const *nd = _get(j);
      if (nd->n_descendants == 1)  // This is only to guard a really obscure case, #284
          nns_dist.emplace_back(D::distance(nd, v_node, _f), j);
    }

    size_t m = nns_dist.size(), p = std::min(m, n);
    if( n < m ) // Has more than N results, so get only top N
      std::partial_sort(nns_dist.begin(), nns_dist.begin() + n, nns_dist.end());
    else
      std::sort(nns_dist.begin(), nns_dist.end());

    // prealloc result buffers
    result->reserve(p);
    if (distances)
    {
        distances->reserve(p);
        for (size_t i = 0; i < p; i++) {
          distances->push_back(D::normalized_distance(nns_dist[i].first));
          result->push_back(nns_dist[i].second);
        }
    }
    else {
      for (size_t i = 0; i < p; i++) {
        result->push_back(nns_dist[i].second);
      }
    }
  }
};

struct EuclideanPacked16 : Euclidean
{
  typedef uint16_t PackedFloatType;
  /* we only redefined distance function to work with normal float data on right(y) side
     and packed vector on left(x) side(on storage?)
  */
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    __builtin_prefetch((uint8_t const*)x + 8);
    T pp = x->norm;
    T qq = y->norm;    
    T pq = decode_and_dot_i16_f32((PackedFloatType const*)x->v, y->v, f);
    return pp + qq - 2*pq;
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return n->a + decode_and_dot_i16_f32((PackedFloatType const*)n->v, y, f);
  }
};


// vim: tabstop=2 shiftwidth=2

