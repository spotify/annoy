#include <stdio.h>
#include <string>
#include <boost/python.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace std;
using namespace boost;

template<typename T>
struct __attribute__((__packed__)) node {
  // We store a binary tree where each node has two things
  // - A vector associated with it
  // - Two children
  // All nodes occupy the same amount of memory
  // All nodes with n_descendants == 1 are leaf nodes.
  // A memory optimization is that for nodes with 2 <= n_descendants <= K,
  // we skip the vector. Instead we store a list of all descendants. K is
  // determined by the number of items that fits in the same space.
  // For nodes with n_descendants == 1 or > K, there is always a
  // corresponding vector. 
  // Note that we can't really do sizeof(node<T>) because we cheat and allocate
  // more memory to be able to fit the vector outside
  int n_descendants;
  int children[2]; // Will possibly store more than 2
  T v[0]; // Hack. We just allocate as much memory as we need and let this array overflow
};

template<typename T>
class AnnoyIndex {
  int _f;
  size_t _s;
  int _n_items;
  mt19937 _rng;
  normal_distribution<T> _nd;
  variate_generator<mt19937&, 
		    normal_distribution<T> > _var_nor;
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  int _n_nodes;
  int _nodes_size;
  vector<int> _roots;
  int _K;
public:
  AnnoyIndex(int f) : _rng(), _nd(), _var_nor(_rng, _nd) {
    _f = f;
    _s = sizeof(node<T>) + sizeof(T) * f; // Size of each node
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _nodes = NULL;

    _K = (sizeof(T) * f + sizeof(int) * 2) / sizeof(int);
    printf("K = %d\n", _K);
  }
  ~AnnoyIndex() {
    if (_nodes)
      free(_nodes);
  }

  void add_item(int item, const python::list& v) {
    _allocate_size(item+1);
    node<T>* n = _get(item);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = python::extract<T>(v[z]);

    if (item >= _n_items)
      _n_items = item + 1;
  }

  void build(int p) {
    _n_nodes = _n_items;
    for (int q = 0; q < p; q++) {
      printf("pass %d...\n", q);

      vector<int> indices;
      for (int i = 0; i < _n_items; i++)
	indices.push_back(i);

      _roots.push_back(_make_tree(indices));
    }
    printf("has %d nodes\n", _n_nodes);
  }

  void save(const string& filename) {
    FILE *f = fopen(filename.c_str(), "w");
    
    fwrite(_nodes, _s, _n_nodes, f);

    fclose(f);

    free(_nodes);
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _nodes = NULL;
    load(filename);
  }

  void load(const string& filename) {
    struct stat buf;
    stat(filename.c_str(), &buf);
    int size = buf.st_size;
    printf("size = %d\n", size);
    int fd = open(filename.c_str(), O_RDONLY, (mode_t)0400);
    printf("fd = %d\n", fd);
    _nodes = (node<T>*)mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);

    int n = size / _s;
    printf("%d nodes\n", n);

    // Find the nodes with the largest number of descendants. These are the roots
    int m = 0;
    for (int i = 0; i < n; i++) {
      if (_get(i)->n_descendants > m) {
	_roots.clear();
	m = _get(i)->n_descendants;
      }
      if (_get(i)->n_descendants == m) {
	_roots.push_back(i);
      }
    }
    printf("found %lu roots with degree %d\n", _roots.size(), m);
  }

  inline T _cos(const T* x, const T* y) {
    T pp = 0, qq = 0, pq = 0;
    for (int z = 0; z < _f; z++) {
      pp += x[z] * x[z];
      qq += y[z] * y[z];
      pq += x[z] * y[z];
    }
    T ppqq = pp * qq;
    if (ppqq > 0) return pq / sqrt(ppqq);
    else return 0.0;
  }

  inline T cos(int i, int j) {
    const T* x = _get(i)->v;
    const T* y = _get(j)->v;
    return _cos(x, y);
  }

  python::list get_nns_by_item(int item, int n) {
    const node<T>* m = _get(item);
    return _get_all_nns(m->v, n);
  }
  python::list get_nns_by_vector(python::list v, int n) {
    vector<T> w(_f);
    for (int z = 0; z < _f; z++)
      w[z] = python::extract<T>(v[z]);
    return _get_all_nns(&w[0], n);
  }
private:
  void _allocate_size(int n) {
    if (n > _nodes_size) {
      int new_nodes_size = (_nodes_size + 1) * 2;
      if (n > new_nodes_size)
	new_nodes_size = n;
      printf("reallocating to %d nodes\n", new_nodes_size);
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset(_nodes + _nodes_size * _s, 0, (new_nodes_size - _nodes_size) * _s);
      _nodes_size = new_nodes_size;
    }
  }

  inline node<T>* _get(int i) {
    return (node<T>*)(_nodes + _s * i);
  }

  int _make_tree(const vector<int >& indices) {
    if (indices.size() == 1)
      return indices[0];

    _allocate_size(_n_nodes + 1);
    int item = _n_nodes++;
    node<T>* m = _get(item);
    m->n_descendants = indices.size();

    if (indices.size() <= _K) {
      for (int i = 0; i < indices.size(); i++)
	m->children[i] = indices[i];
      return item;
    }

    vector<int> children_indices[2];
    for (int attempt = 0; attempt < 20; attempt ++) {
      // Create a random hyperplane
      for (int z = 0; z < _f; z++)
	m->v[z] = _var_nor();
      
      children_indices[0].clear();
      children_indices[1].clear();

      for (int i = 0; i < indices.size(); i++) {
	int j = indices[i];
	node<T>* n = _get(j);
	if (!n) {
	  printf("node %d undef...\n", j);
	  continue;
	}
	T dot = 0;
	for (int z = 0; z < _f; z++) {
	  dot += m->v[z] * n->v[z];
	}
	children_indices[dot > 0].push_back(j);	
      }

      if (children_indices[0].size() > 0 && children_indices[1].size() > 0) {
	if (indices.size() > 100000)
	  printf("Split %lu items -> %lu, %lu (attempt %d)\n", indices.size(), children_indices[0].size(), children_indices[1].size(), attempt);
	break;
      }
    }

    while (children_indices[0].size() == 0 || children_indices[1].size() == 0) {
      // TODO: write vector
      if (indices.size() > 100000)
	printf("Failed splitting %lu items\n", indices.size());

      children_indices[0].clear();
      children_indices[1].clear();

      for (int i = 0; i < indices.size(); i++) {
	int j = indices[i];
	// Just randomize...
	children_indices[_var_nor() > 0].push_back(j);
      }
    }

    int children_0 = _make_tree(children_indices[0]);
    int children_1 = _make_tree(children_indices[1]);

    // We need to fetch m again because it might have been reallocated
    m = _get(item);
    m->children[0] = children_0;
    m->children[1] = children_1;

    return item;
  }

  void _get_nns(const T* v, int i, vector<int>* result, int limit) {
    const node<T>* n = _get(i);

    if (n->n_descendants == 0) {
      // unknown item, nothing to do...
    } else if (n->n_descendants == 1) {
      result->push_back(i);
    } else if (n->n_descendants <= _K) {
      for (int j = 0; j < n->n_descendants; j++) {
	result->push_back(n->children[j]);
      }
    } else {
      T dot = 0;

      for (int z = 0; z < _f; z++) {
	dot += v[z] * n->v[z];
      }
      _get_nns(v, n->children[dot > 0], result, limit);
      if (result->size() < limit)
	_get_nns(v, n->children[dot < 0], result, limit);
    }
  }

  python::list _get_all_nns(const T* v, int n) {
    vector<pair<T, int> > nns_cos;

    for (int i = 0; i < _roots.size(); i++) {
      vector<int> nns;
      _get_nns(v, _roots[i], &nns, n);
      for (int j = 0; j < nns.size(); j++) {
	nns_cos.push_back(make_pair(_cos(v, _get(j)->v), nns[j]));
      }
    }
    sort(nns_cos.begin(), nns_cos.end());
    int last = -1, length=0;
    python::list l;
    for (int i = nns_cos.size() - 1; i >= 0 && length < n; i--) {
      if (nns_cos[i].second != last) {
	l.append(nns_cos[i].second);
	last = nns_cos[i].second;
	length++;
      }
    }
    return l;
  }
};

BOOST_PYTHON_MODULE(annoylib)
{
  python::class_<AnnoyIndex<float> >("AnnoyIndex", python::init<int>())
    .def("add_item",          &AnnoyIndex<float>::add_item)
    .def("build",             &AnnoyIndex<float>::build)
    .def("save",              &AnnoyIndex<float>::save)
    .def("load",              &AnnoyIndex<float>::load)
    .def("cos",               &AnnoyIndex<float>::cos)
    .def("get_nns_by_item",   &AnnoyIndex<float>::get_nns_by_item)
    .def("get_nns_by_vector", &AnnoyIndex<float>::get_nns_by_vector);
}
