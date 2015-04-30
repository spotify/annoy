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
#include "structmember.h"
#include <exception>
#include <stdint.h>


template<typename S, typename T, typename Distance>
class AnnoyIndexPython : public AnnoyIndex<S, T, Distance > {
public:

  AnnoyIndexPython(int f): AnnoyIndex<S, T, Distance>(f) {}

  void add_item_py(S item, const vector<T> w) {
    this->add_item(item, &w[0]);
  }

  PyObject* get_nns_by_item_py(S item, size_t n) {
    PyObject* l = PyList_New(0);
    vector<S> result;
    this->get_nns_by_item(item, n, &result);
    for (size_t i = 0; i < result.size(); i++) {
      PyList_Append(l, PyInt_FromLong(result[i]));
    }
    return l;
  }

  PyObject* get_nns_by_vector_py(PyObject* v, size_t n) {
    vector<T> w(this->_f);
    for (int z = 0; z < PyList_Size(v) && z < this->_f; z++) {
        PyObject *pf = PyList_GetItem(v,z);
        w[z] = PyFloat_AsDouble(pf);
    }
    vector<S> result;
    this->get_nns_by_vector(&w[0], n, &result);
    PyObject* l = PyList_New(0);
    for (size_t i = 0; i < result.size(); i++) {
      PyList_Append(l, PyInt_FromLong(result[i]));
    }
    return l;
  }

  PyObject* get_item_vector_py(S item) {
    const typename Distance::node* m = this->_get(item);
    const T* v = m->v;
    PyObject* l = PyList_New(0);
    for (int z = 0; z < this->_f; z++) {
      PyList_Append(l, PyInt_FromLong(v[z]));
    }
    return l;
  }

  bool save_py(const char* filename) {
    return this->save(filename);
  }

  bool load_py(const char* filename) {
    return this->load(filename);
  }

};


class AI : public AnnoyIndexPython <int32_t, float, Angular<int32_t, float> > {
  public:
  AI(int f) : AnnoyIndexPython <int32_t, float, Angular<int32_t, float> >(f){};
};
class EI : public AnnoyIndexPython <int32_t, float, Euclidean<int32_t, float> > {
  public:
  EI(int f) : AnnoyIndexPython <int32_t, float, Euclidean<int32_t, float> >(f){};
};


// annoy python object
typedef struct {
  PyObject_HEAD
  int f;
  char ch_metric;
  void* ptr;
} py_annoy;


static PyObject *
py_an_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  py_annoy *self;

  self = (py_annoy *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->f = 0;
    self->ch_metric = 0;
    self->ptr = NULL;
  }

  return (PyObject *)self;
}


static int 
py_an_init(py_annoy *self, PyObject *args, PyObject *kwds) {
  const char *metric;

  if (!PyArg_ParseTuple(args, "is", &self->f, &metric))
    return -1;
  self->ch_metric = metric[0];
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = new AI(self->f);
    self->ptr = static_cast<AI*>(static_cast<void*>(ai));
    break;
  case 'e':
    ei = new EI(self->f);
    self->ptr = static_cast<EI*>(static_cast<void*>(ei));
    break;
  }
  return 0;
}


static void 
py_an_dealloc(py_annoy* self) {
  if (self->ptr) {
    switch(self->ch_metric) {
    case 'a':
      delete (AI*)self->ptr;
      break;
    case 'e':
      delete (EI*)self->ptr;
      break;
    }
  }
  self->ob_type->tp_free((PyObject*)self);
}


static PyMemberDef py_annoy_members[] = {
  {(char*)"_f", T_INT, offsetof(py_annoy, f), 0,
   (char*)""},
  {(char*)"ch_metric", T_CHAR, offsetof(py_annoy, ch_metric), 0,
   (char*)"a|e for angular|euclidean"},
  {NULL}	/* Sentinel */
};


static PyObject *
py_an_load(py_annoy *self, PyObject *args) {
  char* filename;
  bool res = false;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "s", &filename))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    res = ai->load_py(filename);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    res = ei->load_py(filename);
    break;
  }
  if (!res) {
    PyErr_SetFromErrno(PyExc_IOError);
    return NULL;
  }
  return PyInt_FromLong(0);
}


static PyObject *
py_an_save(py_annoy *self, PyObject *args) {
  char *filename;
  bool res = false;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "s", &filename))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    res = ai->save_py(filename);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    res = ei->save_py(filename);
    break;
  }
  if (!res) {
    PyErr_SetFromErrno(PyExc_IOError);
    return NULL;
  }
  return PyInt_FromLong(0);
}


static PyObject* 
py_an_get_nns_by_item(py_annoy *self, PyObject *args) {
  PyObject* l = PyList_New(0);
  int32_t item, n;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "ii", &item, &n))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    l = ai->get_nns_by_item_py(item, (size_t)n);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    l = ei->get_nns_by_item_py(item, (size_t)n);
    break;
  }
  return l;
}


static PyObject* 
py_an_get_nns_by_vector(py_annoy *self, PyObject *args) {
  PyObject* v;
  PyObject* l = PyList_New(0);
  int32_t n;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "Oi", &v, &n))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    l = ai->get_nns_by_vector_py(v, (size_t)n);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    l = ei->get_nns_by_vector_py(v, (size_t)n);
    break;
  }
  return l;
}


static PyObject* 
py_an_get_item_vector(py_annoy *self, PyObject *args) {
  PyObject* l = NULL;
  int32_t item;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "i", &item))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    l = ai->get_item_vector_py(item);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    l = ei->get_item_vector_py(item);
    break;
  }
  if (l) return l;
  return Py_None;
}


static PyObject* 
py_an_add_item(py_annoy *self, PyObject *args) {
  vector<float> w;

  //void add_item_py(S item, const vector<T> w) {
  PyObject* l;
  int32_t item;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "iO", &item, &l))
    return Py_None;
  for (int z = 0; z < PyList_Size(l); z++) {
    PyObject *pf = PyList_GetItem(l,z);
    w.push_back(PyFloat_AsDouble(pf));
  }
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    ai->add_item_py(item, w);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    ei->add_item_py(item, w);
    break;
  }
  return PyInt_FromLong(0);
}


static PyObject *
py_an_build(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "i", &q))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    ai->build(q);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    ei->build(q);
    break;
  }
  return PyInt_FromLong(0);
}


static PyObject *
py_an_unload(py_annoy *self, PyObject *args) {
  if (!self->ptr) 
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    ai->unload();
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    ei->unload();
    break;
  }
  return PyInt_FromLong(0);
}


static PyObject *
py_an_get_distance(py_annoy *self, PyObject *args) {
  int32_t i,j;
  double d=0;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "ii", &i, &j))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    d = ai->get_distance(i,j);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    d = ei->get_distance(i,j);
    break;
  }
  return PyFloat_FromDouble(d);
}


static PyObject *
py_an_get_n_items(py_annoy *self, PyObject *args) {
  int32_t n=0;
  bool is_n=false;
  if (!self->ptr) 
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    n = ai->get_n_items();
    is_n = true;
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    n = ei->get_n_items();
    is_n = true;
    break;
  }
  if (is_n) return PyInt_FromLong(n);
  return Py_None;
}


static PyObject *
py_an_verbose(py_annoy *self, PyObject *args) {
  int verbose;
  if (!self->ptr) 
    return Py_None;
  if (!PyArg_ParseTuple(args, "i", &verbose))
    return Py_None;
  AI *ai; EI *ei;
  switch(self->ch_metric) {
  case 'a':
    ai = static_cast<AI*>(self->ptr);
    ai->verbose((bool)verbose);
    break;
  case 'e':
    ei = static_cast<EI*>(self->ptr);
    ei->verbose((bool)verbose);
    break;
  }
  return PyInt_FromLong(0);
}


static PyMethodDef AnnoyMethods[] = {
  {"load",	(PyCFunction)py_an_load, METH_VARARGS, ""},
  {"save",	(PyCFunction)py_an_save, METH_VARARGS, ""},
  {"get_nns_by_item",(PyCFunction)py_an_get_nns_by_item, METH_VARARGS, ""},
  {"get_nns_by_vector",(PyCFunction)py_an_get_nns_by_vector, METH_VARARGS, ""},
  {"get_item_vector",(PyCFunction)py_an_get_item_vector, METH_VARARGS, ""},
  {"add_item",(PyCFunction)py_an_add_item, METH_VARARGS, ""},
  {"build",(PyCFunction)py_an_build, METH_VARARGS, ""},
  {"unload",(PyCFunction)py_an_unload, METH_VARARGS, ""},
  {"get_distance",(PyCFunction)py_an_get_distance, METH_VARARGS, ""},
  {"get_n_items",(PyCFunction)py_an_get_n_items, METH_VARARGS, ""},
  {"verbose",(PyCFunction)py_an_verbose, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};


static PyTypeObject PyAnnoyType = {
  PyObject_HEAD_INIT(NULL)
  0,                      /*ob_size*/
  "annoy.Annoy",          /*tp_name*/
  sizeof(py_annoy),       /*tp_basicsize*/
  0,                      /*tp_itemsize*/
  (destructor)py_an_dealloc, /*tp_dealloc*/
  0,                      /*tp_print*/
  0,                      /*tp_getattr*/
  0,                      /*tp_setattr*/
  0,                      /*tp_compare*/
  0,                      /*tp_repr*/
  0,                      /*tp_as_number*/
  0,                      /*tp_as_sequence*/
  0,                      /*tp_as_mapping*/
  0,                      /*tp_hash */
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "annoy objects",        /* tp_doc */
  0,                      /* tp_traverse */
  0,                      /* tp_clear */
  0,                      /* tp_richcompare */
  0,                      /* tp_weaklistoffset */
  0,                      /* tp_iter */
  0,                      /* tp_iternext */
  AnnoyMethods,           /* tp_methods */
  py_annoy_members,       /* tp_members */
  0,                      /* tp_getset */
  0,                      /* tp_base */
  0,                      /* tp_dict */
  0,                      /* tp_descr_get */
  0,                      /* tp_descr_set */
  0,                      /* tp_dictoffset */
  (initproc)py_an_init,   /* tp_init */
  0,                      /* tp_alloc */
  py_an_new,              /* tp_new */
};

static PyMethodDef module_methods[] = {
  {NULL}	/* Sentinel */
};

PyMODINIT_FUNC initannoylib(void) {
  PyObject *m;

  if (PyType_Ready(&PyAnnoyType) < 0)
    return;

  m = Py_InitModule("annoylib", module_methods);
  if (m == NULL)
    return;

  if (m == NULL)
    return;

  Py_INCREF(&PyAnnoyType);
  PyModule_AddObject(m, "Annoy", (PyObject *)&PyAnnoyType);
}

// vim: tabstop=2 shiftwidth=2
