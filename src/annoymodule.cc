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

#include "annoylib.h"
#include "Python.h"
#include <boost/python.hpp>
#include <exception>
#include <stdint.h>

using namespace std;
using namespace boost;


struct ErrnoException : std::exception {};

void TranslateException(ErrnoException const& e) {
  PyErr_SetFromErrno(PyExc_IOError);
}

template<typename S, typename T, typename Distance>
class AnnoyIndexPython : public AnnoyIndex<S, T, Distance > {
public:
  AnnoyIndexPython(int f): AnnoyIndex<S, T, Distance>(f) {}
  void add_item_py(S item, const python::list& v, size_t label) {
    vector<T> w;
    for (int z = 0; z < this->_f; z++)
      w.push_back(python::extract<T>(v[z]));

    this->add_item(item, &w[0], label);
  }
  S set_K_py(S K) { 
    return this->set_K(K); 
  }
  S set_item_size_py(size_t size) { 
    return this->set_item_size(size); 
  }
  S add_item_to_index_py(const python::list& v, size_t label) {
    vector<T> w;
    for (int z = 0; z < this->_f; z++)
      w.push_back(python::extract<T>(v[z]));

   return this->add_item_to_index(&w[0], label);
  }

  void get_all_groups_py(T dist_threshold) {
    this->get_all_groups(dist_threshold);
    return;
  }

  python::list get_nns_by_item_py(S item, size_t n, python::list c, size_t tn) {
    size_t c_size = boost::python::len(c);
    vector<S> w(c_size);
    for (int z = 0; z < c_size; z++)
      w[z] = python::extract<S>(c[z]);
    vector<pair<T, S> > result;
    this->get_nns_by_item(item, n, &result, w, tn);
    python::list l;
    for (size_t i = 0; i < result.size(); i++) {
      python::list t;
      t.append(result[i].first);
      t.append(result[i].second);
      l.append(t);
    }
    return l;
  }

  python::list get_nns_group_by_item_py(S item, size_t n, python::list c, size_t tn, T dist_threshold) {
    size_t c_size = boost::python::len(c);
    vector<S> cs(c_size);
    vector<vector<S> > group_result;
    this->get_nns_group_by_item(item, n, &group_result, cs, tn, dist_threshold);
    python::list l;
    for (size_t i = 0; i < group_result.size(); i++) {
      python::list a;
      for (size_t j = 0; j < group_result[i].size(); j ++ ) { 
        a.append(group_result[i][j]); 
      }
      l.append(a);
    }
    return l;
  }

  python::list get_nns_group_by_vector_py(python::list v, size_t n, python::list c, size_t tn, T dist_threshold) {
    size_t c_size = boost::python::len(c);
    vector<S> cs(c_size);
    for (int z = 0; z < c_size; z++)
      cs[z] = python::extract<S>(c[z]);
    vector<T> w(this->_f);
    for (int z = 0; z < this->_f; z++)
      w[z] = python::extract<T>(v[z]);
    vector<vector<S> > group_result;
    this->get_nns_group_by_vector(&w[0], n, &group_result, cs, tn, dist_threshold);
    python::list l;
    for (size_t i = 0; i < group_result.size(); i++) {
      python::list a;
      for (size_t j = 0; j < group_result[i].size(); j ++ ) { 
        a.append(group_result[i][j]); 
      }
      l.append(a);
    }
    return l;
  }
  python::list get_nns_by_vector_py(python::list v, size_t n, python::list c, size_t tn) {
    size_t c_size = boost::python::len(c);
    vector<S> cs(c_size);
    for (int z = 0; z < c_size; z++)
      cs[z] = python::extract<S>(c[z]);
    vector<T> w(this->_f);
    for (int z = 0; z < this->_f; z++)
      w[z] = python::extract<T>(v[z]);
    vector<pair<T, S> > result;
    this->get_nns_by_vector(&w[0], n, &result, cs, tn);
    python::list l;
    for (size_t i = 0; i < result.size(); i++) {
      python::list t;
      t.append(result[i].first);
      t.append(result[i].second);
      l.append(t);
    }
    return l;
  }

  python::list get_item_vector_py(S item) {
    const typename Distance::node* m = this->_get(item);
    const T* v = m->v;
    python::list l;
    for (int z = 0; z < this->_f; z++) {
      l.append(v[z]);
    }
    return l;
  }

  void save_py(const string& filename) {
    if (!this->save(filename))
      throw ErrnoException();
  }
  void load_py(const string& filename) {
    if (!this->load(filename))
      throw ErrnoException();
  }
  void load_memory_py(const string& filename) {
    if (!this->load_memory(filename))
      throw ErrnoException();
  }
};

template<typename C>
void expose_methods(python::class_<C> c) {
  c.def("add_item",          &C::add_item_py)
    .def("add_item_to_index", &C::add_item_to_index_py)
    .def("set_item_size",     &C::set_item_size)
    .def("set_K",    	      &C::set_K_py)
    .def("build",             &C::build)
    .def("save",              &C::save_py)
    .def("load",              &C::load_py)
    .def("load_memory",       &C::load_memory_py)
    .def("unload",            &C::unload)
    .def("get_distance",      &C::get_distance)
    .def("get_all_groups",    &C::get_all_groups_py)
    .def("get_nns_by_item",   &C::get_nns_by_item_py)
    .def("get_nns_group_by_item",     &C::get_nns_group_by_item_py)
    .def("get_nns_group_by_vector",     &C::get_nns_group_by_vector_py)
    .def("get_nns_by_vector", &C::get_nns_by_vector_py)
    .def("get_item_vector",   &C::get_item_vector_py)
    .def("get_n_items",       &C::get_n_items)
    .def("verbose",           &C::verbose);

}

BOOST_PYTHON_MODULE(annoylib)
{
  python::register_exception_translator<ErrnoException>(&TranslateException);
  expose_methods(python::class_<AnnoyIndexPython<int32_t, float, Angular<int32_t, float> > >("AnnoyIndexAngular", python::init<int>()));
  expose_methods(python::class_<AnnoyIndexPython<int32_t, float, Euclidean<int32_t, float> > >("AnnoyIndexEuclidean", python::init<int>()));
}
