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

#pragma once

#include "packutils.h"
#include "annoylib.h"
#include "datamapper.h"
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <iostream>
#include <alloca.h>
#include <assert.h>

#ifdef __GNUC__
#  define alloca_aligned(sz) __builtin_alloca_with_align(sz, 64)
#else
/* Clang must be generated already aligned stack allocation */
#  define alloca_aligned(sz) alloca(sz)
#endif

namespace Annoy {

namespace detail {
  // This is common storage header(tailer)
  struct Header
  {
    uint32_t version;
    uint32_t vlen, idx_block_len;
    uint32_t nblocks;
  };

  static_assert( sizeof(Header) == 16, "header must 16 bytes long!");


  class FileWriter
  {
  public:
    ~FileWriter()
    {
      if( f )
        fclose(f);
    }
    bool open( char const *filename, size_t calculated_size )
    {
      f = fopen(filename, "wb");
      if (f == nullptr)
        return false;

      return true;
    }
    void write( void const *buf, size_t sz, size_t cnt )
    {
      fwrite(buf, sz, cnt, f);
    }
  private:
    FILE *f = nullptr;
  };

  class MMapWriter
  {
  public:
    typedef DataMapping Mapping;
  public:
    ~MMapWriter() { destroy(); }
    MMapWriter() = default;
    MMapWriter( MMapWriter const& ) = delete;
    MMapWriter( MMapWriter && o )
      : maping(o.maping)
      , top(static_cast<uint8_t*>(o.maping))
      , size(o.size)
      , mlocked(o.mlocked)
    {
        o.maping = nullptr;
    }

    MMapWriter& operator == ( MMapWriter && ) = delete;
    MMapWriter& operator == ( MMapWriter const & ) = delete;
    bool open( char const */*filename*/, size_t calculated_size )
    {

      void *p = calculated_size ? mmap(0, calculated_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, 0, 0)
                                : nullptr;
      if( p == MAP_FAILED )
        return false;

      maping = p;
      top = static_cast<uint8_t*>(p);
      size = calculated_size;

#if defined(MADV_DONTDUMP)
      // Exclude from a core dump those pages
      if (p != nullptr)
        madvise(p, calculated_size, MADV_DONTDUMP);
#endif
      return true;
    }

    void write( void const *buf, size_t sz, size_t cnt )
    {
      size_t total = sz * cnt;
      memcpy(top, buf, total);
      top += total;
    }

    DataMapping map( char const */*filename*/, bool need_mlock ) {
      if (need_mlock) {
        mlock(maping, size);
        mlocked = true;
      }
      return DataMapping{maping, size};
    }

    void unmap( DataMapping const & ) {
      destroy();
    }

    void* get_ptr() const { return maping; }
    size_t get_size() const { return size; }
    
    Mapping clone( Mapping m ) const {
      return clone_mmap(m, MAP_ANONYMOUS | MAP_PRIVATE);
    }
  private:
    void destroy() {
      if( maping ) {
        //if( mlocked )
        //  munlock(maping, size);
        munmap(maping, size);
        maping = nullptr;
      }
    }
  private:
    void *maping = nullptr;
    uint8_t *top = nullptr;
    size_t size = 0;
    bool mlocked = false;
  };

} // namespace detail

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
  // this flag is used for node_id to separate indices only nodes and original
  static constexpr S const _K_mask = S(1) << S(sizeof(S) * 8 - 1);

protected:
  const int _f;
  S  _s; // Size of each node
  S const _K; // Max number of descendants to fit into node
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  vector<S> _roots;
  S _n_items;
  size_t _nodes_size; // capacity of the write buffer
  S _n_nodes; // number of nodes
  Random _random;
  bool _verbose;

  typedef std::vector<S> indices_list_t;
  std::deque<indices_list_t>  _indices_lists;
public:
  PackedAnnoyIndexer(int f) : PackedAnnoyIndexer(f, f) {}
  PackedAnnoyIndexer(int f, S idx_block_len)
    : _f(f)
    , _s(offsetof(Node, v) + _f * sizeof(T))
    , _K(idx_block_len)
    , _verbose(false)
  {
    if( uint32_t(f) % 8 )
      throw std::runtime_error("number of element in the vector must be multiply of 8.");
    if( (_K * sizeof(S)) % 16 )
      throw std::runtime_error("size of the index-node must be multiply of 16 bytes, consider using different idx_block_len!");
    if( _K > S(f) )
      throw std::runtime_error("size of index-node cannot be greater than vector length!");

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
    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;
    while (1) {
      if (q == -1)
      {
        if (_n_nodes >= _n_items * 2)
          break;
      }
      else if (_roots.size() >= (size_t)q)
        break;

      if (_verbose) annoylib_showUpdate("pass %zd...\n", _roots.size());


      vector<S> indices;
      for (S i = 0; i < _n_items; i++) {
          if (_get(i)->n_descendants >= 1) // Issue #223
          indices.push_back(i);
      }

      // cannot make roots w/o items
      if( indices.empty() )
        break;

      _roots.push_back(_make_tree(indices, true));
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

    if (_verbose) annoylib_showUpdate("has %d nodes\n", _n_nodes);
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

  template<typename FWriter>
  bool save_impl(FWriter &w, const char* filename) {
    // calc size of packed node
    size_t packed_size = offsetof(Node, v) + _f * sizeof(uint16_t);
    size_t const iblocks = _indices_lists.size();

    if (_verbose) {
      // get indices stats?
      uint32_t min_max_bit = 100, max_max_bit = 0;
      size_t total_bits = 0, total_size = 0;
      for( auto const &i : _indices_lists )
      {
        total_size += i.size();
        uint32_t const *idx = reinterpret_cast<uint32_t const*>(i.data()), *e = idx + i.size();
        uint32_t mb = get_max_bits(idx, e);
        min_max_bit = std::min(min_max_bit, mb);
        max_max_bit = std::max(max_max_bit, mb);
        total_bits += mb;
      }
      annoylib_showUpdate("after pack stats\ntotal normal=%d total_nodes=%d\n"
                 "total size of indices=%zd numbers of blocks=%zd\n"
                 "total number of maxbits=%zd\n",
                 _n_items, _n_nodes, iblocks * _K * sizeof(S), iblocks, total_bits);

      if( iblocks ) {
        auto iblock_avg_sz = total_size / double(iblocks);
        (void)iblock_avg_sz;
        annoylib_showUpdate("iblock stats sizes: avg=%.03f max=%d waste=%.03f %%\n", iblock_avg_sz, _K, (1.0 - (iblock_avg_sz / (_K - 1 ))) * 100.);
      }
    }

    size_t calculated_size = packed_size * _n_nodes // packed vectors size
                             + sizeof(S) * _K * _indices_lists.size() // unpacked indices size
                             + sizeof(detail::Header)
                             ;
    if( !w.open(filename, calculated_size) )
      return false;

    // write indices first
    S index_write_block[_K];
    for( auto const &i : _indices_lists )
    {
      size_t const isz = i.size();
      index_write_block[0] = isz;
      S *data_start = index_write_block + 1;
      memset(data_start, 0, sizeof(index_write_block) - sizeof(S));
      memcpy(data_start, i.data(), isz * sizeof(S));
      if( isz < _K - 1 )
      {
        data_start += isz;
        // zeroing bytes left for stability
        memset(data_start, 0, sizeof(index_write_block) - ((isz + 1) * sizeof(S)));
      }
      w.write(index_write_block, sizeof(index_write_block), 1);
    }
    // and nodes data compress node data one by one
    // allocate tmp buffer
    void *packed_nodes = (Node*)alloca(packed_size);
    for (S i = 0; i < _n_nodes; i++) {
      Node *node = get_node_ptr<S, Node>(_nodes, _s, i);
      Node *packed = get_node_ptr<S, Node>(packed_nodes, packed_size, 0);
      // pack and copy to new buffer
      memcpy(packed, node, offsetof(Node, v));
      pack_float_vector_i16(node->v, (uint16_t*)packed->v, _f);
      w.write(packed_nodes, packed_size, 1);
    }
    // header goes to the tail
    // write header only at tail of file to keep strict alignment in memory
    // for much faster memory access
    detail::Header hdr;
    hdr.version = 0;
    hdr.vlen = _f;
    hdr.idx_block_len = _K;
    hdr.nblocks = iblocks;
    w.write(&hdr, sizeof(hdr), 1);

    unload();

    return true;
  }

  bool save(const char* filename) {
    detail::FileWriter wr;
    return save_impl(wr, filename);
  }

  void reinitialize() {
    _nodes = nullptr;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _roots.clear();
    _indices_lists.clear();
  }

  void unload() {
    if (_nodes) {
      // We have heap allocated data
      free(_nodes);
    }
    reinitialize();
    if (_verbose) annoylib_showUpdate("unloaded\n");
  }

  void set_seed(int seed) {
    _random.set_seed(seed);
  }

  void verbose(bool v) {
    _verbose = v;
  }

  void preallocate(size_t n) {
    _allocate_size(n);
  }

protected:
  void _allocate_size(size_t n) {
    if (n > _nodes_size) {
      const double reallocation_factor = 1.3;
      size_t new_nodes_size = std::max(n,
                  (size_t)((_nodes_size + 1) * reallocation_factor));
      if (_verbose) annoylib_showUpdate("Reallocating to %zd nodes\n", new_nodes_size);
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset((char *)_nodes + (_nodes_size * _s)/sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
      _nodes_size = new_nodes_size;
    }
  }


  inline Node* _get(S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  S _append_indices( const vector<S>& indices )
  {
    S i = _indices_lists.size();
    _indices_lists.emplace_back( std::move(indices) );
    return i | _K_mask;
  }

  static double _split_imbalance(const vector<S>& left_indices, const vector<S>& right_indices) {
    double ls = (double)left_indices.size();
    double rs = (double)right_indices.size();
    double f = ls / (ls + rs + 1e-9);  // Avoid 0/0
    return std::max(f, 1.0 - f);
  }

  S _make_tree(const vector<S >& indices, bool is_root) {
    // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
    // There's some regrettable complications caused by the problem that root nodes have to be "special":
    // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
    // 2. Root nodes with only 1 child need to be a "dummy" parent
    // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K

    S const isz = indices.size();

    if (isz == 1 && !is_root)
      return indices[0];
    // NOTE: this is very important thing, since we operates only with aligned blocks
    // we cannot store more than _K - 1 indices, first element in the block is number of indices
    S const max_n_descendants = _K - 1;

    if (isz <= max_n_descendants && (!is_root || (size_t)_n_items <= (size_t)max_n_descendants || isz == 1)) {
      if (!is_root)
        // only non-roots can have indices only nodes!
        return _append_indices(indices);

      _allocate_size(_n_nodes + 1);
      S item = _n_nodes++;
      Node* m = _get(item);
      m->n_descendants = _n_items;

      // Using std::copy instead of a loop seems to resolve issues #3 and #13,
      // probably because gcc 4.8 goes overboard with optimizations.
      // Using memcpy instead of std::copy for MSVC compatibility. #235
      // Only copy when necessary to avoid crash in MSVC 9. #293
      if (!indices.empty())
        memcpy(m->children, &indices[0], isz * sizeof(S));

      return item;
    }

    vector<Node*> children;
    for (S i = 0; i < isz; i++) {
      S j = indices[i];
      Node* n = _get(j);
      if (n)
        children.push_back(n);
    }

    vector<S> children_indices[2];
    Node* m = (Node*)alloca(_s);

    for (int attempt = 0; attempt < 3; attempt++) {
      children_indices[0].clear();
      children_indices[1].clear();
      D::create_split(children, _f, _s, _random, m);

      for (S i = 0; i < isz; i++) {
        S j = indices[i];
        Node* n = _get(j);
        if (n) {
          bool side = D::side(m, n->v, _f, _random);
          children_indices[side].push_back(j);
        } else {
          annoylib_showUpdate("No node for index %d?\n", j);
        }
      }

      if (_split_imbalance(children_indices[0], children_indices[1]) < 0.95)
        break;
    }

    // If we didn't find a hyperplane, just randomize sides as a last option
    while (_split_imbalance(children_indices[0], children_indices[1]) > 0.99) {
      if (_verbose)
        annoylib_showUpdate("\tNo hyperplane found (left has %ld children, right has %ld children)\n",
          children_indices[0].size(), children_indices[1].size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      for (int z = 0; z < _f; z++)
        m->v[z] = 0;

      for (S i = 0; i < isz; i++) {
        S j = indices[i];
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());

    m->n_descendants = is_root ? _n_items : (S)isz;
    for (int side = 0; side < 2; side++) {
      // run _make_tree for the smallest child first (for cache locality)
      m->children[side^flip] = _make_tree(children_indices[side^flip], false);
    }

    _allocate_size(_n_nodes + 1);
    S item = _n_nodes++;

    memcpy(_get(item), m, _s);

    return item;
  }
};


template<typename S, typename T, typename Distance, typename DataMapper = MMapDataMapper>
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
  typedef typename D::template Node<S, T> Node;
  typedef std::unique_ptr<Node, decltype( &free )> NodeUPtrType;
  typedef typename DataMapper::Mapping DataMapping;
private:
  static constexpr S const _K_mask = S(1UL) << S(sizeof(S) * 8 - 1);
  static constexpr S const _K_mask_clear = _K_mask - 1;
protected:
  typedef std::pair<T, S>               qpair_t;
  typedef std::vector<qpair_t>          queue_t;
protected:
  int _f;
  S _s; // Size of each node
  S _K; // Max number of descendants to fit into node
  S _n_items; // number of ordinal nodes(i.e leaf)
  void const *_nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  queue_t _roots_q;
  DataMapper  _mapper;
  DataMapping _mapping;
public:

  PackedAnnoySearcher(DataMapper && mapper = DataMapper())
    : _f(0)
    , _s(0)
    , _K(0)
    , _n_items(0)
    , _nodes(nullptr)
    , _mapper(std::move(mapper))
  {
    // check size of node must be multiply of 16
    if( _s % 16 )
      throw std::runtime_error("size of the node must be multiply of 16 bytes, consider using different config!");
  }

  ~PackedAnnoySearcher()
  {
    _mapper.unmap(_mapping);
  }

  // WARNING!
  // this is not a regular clone function
  // this function completely clone memory storages
  // this can be useful to avoid memory bank conflicts and offcore-loads
  // but it's too dangerous to regular user, so don't use me if you not Jedi, sorry...
  PackedAnnoySearcher* clone() const
  {
    std::unique_ptr<PackedAnnoySearcher> n{ new PackedAnnoySearcher() };
    if( nullptr == _nodes )
      throw std::runtime_error("index must be loaded!");

    n->_f = _f;
    n->_s = _s;
    n->_K = _K;
    n->_n_items = _n_items;
    n->_roots_q = _roots_q;

    n->_mapping = _mapper.clone(_mapping);

    if( nullptr == n->_mapping.data )
      throw std::runtime_error("failed to clone mapper data");

    // do proper offset for nodes
    n->_nodes = (char*)n->_mapping.data + size_t((char const*)_nodes - (char const*)_mapping.data);

    return n.release();
  }

  void get_item(S item, T* v) const {
    Node* m = _get(item);
    decode_vector_i16_f32((uint16_t const*)m->v, v, _f);
  }

  bool load(const char* filename, bool need_mlock) {
    _mapping = _mapper.map(filename, need_mlock);
    if (!_mapping.data) {
      return false;
    }

    S const *index_start = static_cast<S const*>(_mapping.data);

    // read pseudo-header
    S nindices = _init_header();

    // get offset to the vector data start
    _nodes = (void*)(index_start + nindices);

    size_t sizeof_indices = nindices * sizeof(S);

    size_t n_nodes = (S)((_mapping.size - sizeof_indices) / _s);



    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    std::vector<S> roots;
    roots.clear();
    S m = -1;
    // WARN: using here forced signed type(long) to eliminate bug with S == unsigned type and empty base
    for (long i = n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == S(-1) || k == m) {
        roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (roots.size() > 1 && _get(roots.front())->children[0] == _get(roots.back())->children[0])
      roots.pop_back();

    _roots_q.reserve(roots.size());

    // convert ordinal roots refs into search queue to reduce time of building heap for scan
    for ( S r : roots ) {
      _roots_q.emplace_back(Distance::template pq_initial_value<T>(), r);
    }
    std::make_heap(_roots_q.begin(), _roots_q.end());
    _n_items = m != -1 ? m : 0;

    return true;
  }

  // this function useful in several cases:
  // 1. exclude huge coredump via MADV_DONTDUMP.
  // 2. preload from storage into memory via MADV_WILLNEED.
  // 3. use THP if you disable it on your system via MADV_HUGEPAGE.
  // 4. something special ;)
  bool madvise(int flags) const {
      return ::madvise(const_cast<void*>(_mapping.data), _mapping.size, flags) == 0;
  }

  T get_distance(S i, S j) const {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    T *mv = static_cast<T*>(alloca_aligned(_f * sizeof(T)));
    const Node* m = _get(item);
    decode_vector_i16_f32((PackedFloatType const*)m->v, mv, _f);
    get_nns_by_vector(mv, n, search_k, result, distances);
  }

  static Node* mk_node( const T* v, int n, void *node_alloc_buf ) {
    Node* v_node = static_cast<Node*>(node_alloc_buf);

    D::template zero_value<Node>(v_node);
    memcpy(v_node->v, v, sizeof(T) * n);
    D::init_node(v_node, n);

    return v_node;
  }

  static NodeUPtrType mk_node_ptr( const T* v, int n ) {
    Node* v_node = mk_node(v, n, malloc(offsetof(Node, v) + n * sizeof(T)));
    return NodeUPtrType(v_node, free);
  }

  void get_nns_by_vector(const T* v, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    // init node for comparison from given vector(v)
    // alloc space for that node on the stack!
    // but node vector(v[1]) data is need to be aligned to 16 bytes
    void *node_alloc_buf = alloca_aligned(offsetof(Node, v) + _f * sizeof(T));
    Node* v_node = mk_node(v, _f, node_alloc_buf);

    vector<pair<T, S> > nns_dist;
    _get_all_nns(v_node, n, search_k, nns_dist, _just_bypass);
    size_t p = std::min(nns_dist.size(), n);
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

  // faster versions of above get_nns() functions
  // supported user-defined filtering
  // WARN: this version return raw-distance, you can normalize it in filter function!
  template<typename Filter>
  void get_nns_by_vector_filter(const T* v, size_t n, size_t search_k, Filter filter, vector<pair<T, S> > &nns_dist) const {
    // init node for comparison from given vector(v)
    // alloc space for that node on the stack!
    // but node vector(v[1]) data is need to be aligned to 16 bytes
    void *node_alloc_buf = alloca_aligned(offsetof(Node, v) + _f * sizeof(T));
    Node* v_node = mk_node(v, _f, node_alloc_buf);

    _get_all_nns(v_node, n, search_k, nns_dist, filter);
  }

  // same as above but referenced himself
  template<typename Filter>
  void get_nns_by_item_filter(S item, size_t n, size_t search_k, Filter filter, vector<pair<T, S> > &nns_dist) const {
    T *mv = static_cast<T*>(alloca_aligned(_f * sizeof(T)));
    const Node* m = _get(item);
    decode_vector_i16_f32((PackedFloatType const*)m->v, mv, _f);
    get_nns_by_vector_filter(mv, n, search_k, filter, nns_dist);
  }

  S get_n_items() const {
    return _n_items;
  }

private:

  S _init_header() {
    detail::Header const *hdr = reinterpret_cast<detail::Header const*>((uint8_t const*)_mapping.data
      + _mapping.size - sizeof(detail::Header));

    _f = hdr->vlen;
    _s = offsetof(Node, v) + _f * sizeof(PackedFloatType);
    _K = hdr->idx_block_len;

    return _K * hdr->nblocks;
  }

  inline Node* _get(S i) const {
    Node *nd = get_node_ptr<S, Node>(_nodes, _s, i);
    __builtin_prefetch(nd);
    return nd;
  }

  static bool _just_bypass( T dist ) {
      return true;
  }

  template<typename Filter>
  void _get_all_nns(const Node* v_node, size_t n, S search_k, vector<pair<T, S> > &nns_dist,
                    Filter filter) const {
    if (search_k == (S)-1)
      search_k = n * _roots_q.size(); // slightly arbitrary default value
    // alloc node-ids temporary search buffer on the heap
    // TODO: this is little faster than using stack,
    // but consider using preallocated memory buffer instead!
    std::unique_ptr<S[]> nns( new S[search_k + _K * 2]);
    // copy prepared queue with roots
    queue_t q;
    // reduce realloc overhead
    // TODO: dTLB high pressure during scan loop, so decide to use HP for temp and output buffers!
    q.reserve( n * _roots_q.size() );
    q.assign(_roots_q.begin(), _roots_q.end());
    S nns_cnt = 0;
    // collect candidates ID's w/o collecting weights to reduce bandwidth penalty
    // due to dups in candidates
    while( !q.empty() ) {
      const pair<T, S>& top = q.front();
      T d = top.first;
      S fi = top.second, i = fi & _K_mask_clear;
      std::pop_heap(q.begin(), q.end());
      q.pop_back();

      if( !(fi & _K_mask) )
      {
        // ordinal node
        Node* nd = _get(i);
        if (nd->n_descendants == 1 && i < _n_items) {
          nns[nns_cnt++] = i;
        } else {
          // split node
          T margin = D::margin(nd, v_node->v, _f);
          q.emplace_back(D::pq_distance(d, margin, 1), S(nd->children[1]));
          std::push_heap(q.begin(), q.end());
          q.emplace_back(D::pq_distance(d, margin, 0), S(nd->children[0]));
          std::push_heap(q.begin(), q.end());
        }
      }
      else
      {
        // index only node
        S const *idx = (S const*)_mapping.data + i * _K;
        __builtin_prefetch(idx);
        S const *dst = idx + 1;
        void *dest = &nns[nns_cnt];
        nns_cnt += *idx;
        memcpy(dest, dst, *idx * sizeof(S));
      }
      if( nns_cnt >= search_k )
        break;
    }

    // sort by ID to eliminate dups
    std::sort(nns.get(), nns.get() + nns_cnt);
    nns_dist.reserve(nns_cnt);

    S last = -1;
    // eliminate dups and calc distances
    for (S i = 0, j; i < nns_cnt; ++i) {
      j = nns[i];
      if (j == last)
        continue;
      last = j;
      Node const *nd = _get(j);
      if (nd->n_descendants == 1) { // This is only to guard a really obscure case, #284
        T dist = D::distance(nd, v_node, _f);
        if( filter(dist) )
          nns_dist.emplace_back(dist, j);
      }
    }

    // resort by distance
    size_t m = nns_dist.size();
    if( n < m ) // Has more than N results, so get only top N
      std::partial_sort(nns_dist.begin(), nns_dist.begin() + n, nns_dist.end());
    else
      std::sort(nns_dist.begin(), nns_dist.end());

  }
};

struct EuclideanPacked16 : Euclidean
{
  using UnpackedT = Euclidean;
  typedef uint16_t PackedFloatType;
  /* we only redefined distance function to work with normal float data on right(y) side
     and packed vector on left(x) side(on storage?)
  */
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return decode_and_euclidean_distance_i16_f32((PackedFloatType const*)x->v, y->v, f);
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return n->a + decode_and_dot_i16_f32((PackedFloatType const*)n->v, y, f);
  }
};

struct DotProductPacked16 : DotProduct
{
  using UnpackedT = DotProduct;
  typedef uint16_t PackedFloatType;
  /* we only redefined distance function to work with normal float data on right(y) side
     and packed vector on left(x) side(on storage?)
  */
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return -decode_and_dot_i16_f32((PackedFloatType const*)x->v, y->v, f);
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return decode_and_dot_i16_f32((PackedFloatType const*)n->v, y, f) + (n->dot_factor * n->dot_factor);
  }
};

} // namespace Annoy

#undef alloca_aligned

// vim: tabstop=2 shiftwidth=2

