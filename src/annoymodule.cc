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
#include "kissrandom.h"
#include "Python.h"
#include "structmember.h"
#include <exception>
#include <stdint.h>


#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#ifdef IS_PY3K
    #define PyInt_FromLong PyLong_FromLong 
#endif


template class AnnoyIndexInterface<int32_t, float>;

// annoy python object
typedef struct {
  PyObject_HEAD
  int f;
  AnnoyIndexInterface<int32_t, float>* ptr;
} py_annoy;


static PyObject *
py_an_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  py_annoy *self;

  self = (py_annoy *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->f = 0;
    self->ptr = NULL;
  }

  return (PyObject *)self;
}


static int 
py_an_init(py_annoy *self, PyObject *args, PyObject *kwds) {
  const char *metric;

  if (!PyArg_ParseTuple(args, "is", &self->f, &metric))
    return -1;
  switch(metric[0]) {
  case 'a':
    self->ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random>(self->f);
    break;
  case 'e':
    self->ptr = new AnnoyIndex<int32_t, float, Euclidean, Kiss64Random>(self->f);
    break;
  }
  return 0;
}


static void 
py_an_dealloc(py_annoy* self) {
  if (self->ptr) {
    delete self->ptr;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyMemberDef py_annoy_members[] = {
  {(char*)"_f", T_INT, offsetof(py_annoy, f), 0,
   (char*)""},
  {NULL}	/* Sentinel */
};


static PyObject *
py_an_load(py_annoy *self, PyObject *args) {
  char* filename;
  bool res = false;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "s", &filename))
    Py_RETURN_NONE;

  res = self->ptr->load(filename);

  if (!res) {
    PyErr_SetFromErrno(PyExc_IOError);
    return NULL;
  }
  Py_RETURN_TRUE;
}


static PyObject *
py_an_save(py_annoy *self, PyObject *args) {
  char *filename;
  bool res = false;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "s", &filename))
    Py_RETURN_NONE;

  res = self->ptr->save(filename);

  if (!res) {
    PyErr_SetFromErrno(PyExc_IOError);
    return NULL;
  }
  Py_RETURN_TRUE;
}


PyObject*
get_nns_to_python(const vector<int32_t>& result, const vector<float>& distances, int include_distances) {
  PyObject* l = PyList_New(result.size());
  for (size_t i = 0; i < result.size(); i++)
    PyList_SetItem(l, i, PyInt_FromLong(result[i]));
  if (!include_distances)
    return l;

  PyObject* d = PyList_New(distances.size());
  for (size_t i = 0; i < distances.size(); i++)
    PyList_SetItem(d, i, PyFloat_FromDouble(distances[i]));

  PyObject* t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, l);
  PyTuple_SetItem(t, 1, d);

  return t;
}


static PyObject* 
py_an_get_nns_by_item(py_annoy *self, PyObject *args) {
  int32_t item, n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "ii|ii", &item, &n, &search_k, &include_distances))
    Py_RETURN_NONE;

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_item(item, n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


static PyObject* 
py_an_get_nns_by_vector(py_annoy *self, PyObject *args) {
  PyObject* v;
  int32_t n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "Oi|ii", &v, &n, &search_k, &include_distances))
    Py_RETURN_NONE;

  vector<float> w(self->f);
  for (int z = 0; z < PyList_Size(v) && z < self->f; z++) {
    PyObject *pf = PyList_GetItem(v,z);
    w[z] = PyFloat_AsDouble(pf);
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_vector(&w[0], n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


static PyObject* 
py_an_get_item_vector(py_annoy *self, PyObject *args) {
  int32_t item;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "i", &item))
    Py_RETURN_NONE;

  vector<float> v(self->f);
  self->ptr->get_item(item, &v[0]);
  PyObject* l = PyList_New(self->f);
  for (int z = 0; z < self->f; z++) {
    PyList_SetItem(l, z, PyFloat_FromDouble(v[z]));
  }

  return l;
}


static PyObject* 
py_an_add_item(py_annoy *self, PyObject *args) {
  PyObject* l;
  int32_t item;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "iO", &item, &l))
    Py_RETURN_NONE;

  vector<float> w(self->f, 0.0);
  for (int z = 0; z < self->f; z++) {
    PyObject *pf = PyList_GetItem(l,z);
    w[z] = PyFloat_AsDouble(pf);
  }
  self->ptr->add_item(item, &w[0]);

  Py_RETURN_NONE;
}


static PyObject *
py_an_build(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "i", &q))
    Py_RETURN_NONE;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->build(q);
  Py_END_ALLOW_THREADS;

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unbuild(py_annoy *self, PyObject *args) {
  if (!self->ptr) 
    Py_RETURN_NONE;
  
  Py_BEGIN_ALLOW_THREADS;
  self->ptr->unbuild();
  Py_END_ALLOW_THREADS;

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unload(py_annoy *self, PyObject *args) {
  if (!self->ptr) 
    Py_RETURN_NONE;

  self->ptr->unload();

  Py_RETURN_TRUE;
}


static PyObject *
py_an_get_distance(py_annoy *self, PyObject *args) {
  int32_t i, j;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "ii", &i, &j))
    Py_RETURN_NONE;

  double d = self->ptr->get_distance(i,j);
  return PyFloat_FromDouble(d);
}


static PyObject *
py_an_get_n_items(py_annoy *self, PyObject *args) {
  if (!self->ptr) 
    Py_RETURN_NONE;

  int32_t n = self->ptr->get_n_items();
  return PyInt_FromLong(n);
}


static PyObject *
py_an_verbose(py_annoy *self, PyObject *args) {
  int verbose;
  if (!self->ptr) 
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "i", &verbose))
    Py_RETURN_NONE;

  self->ptr->verbose((bool)verbose);

  Py_RETURN_TRUE;
}


static PyObject *
py_an_set_seed(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr)
    Py_RETURN_NONE;
  if (!PyArg_ParseTuple(args, "i", &q))
    Py_RETURN_NONE;

  self->ptr->set_seed(q);

  Py_RETURN_NONE;
}


static PyMethodDef AnnoyMethods[] = {
  {"load",	(PyCFunction)py_an_load, METH_VARARGS, ""},
  {"save",	(PyCFunction)py_an_save, METH_VARARGS, ""},
  {"get_nns_by_item",(PyCFunction)py_an_get_nns_by_item, METH_VARARGS, ""},
  {"get_nns_by_vector",(PyCFunction)py_an_get_nns_by_vector, METH_VARARGS, ""},
  {"get_item_vector",(PyCFunction)py_an_get_item_vector, METH_VARARGS, ""},
  {"add_item",(PyCFunction)py_an_add_item, METH_VARARGS, ""},
  {"build",(PyCFunction)py_an_build, METH_VARARGS, ""},
  {"unbuild",(PyCFunction)py_an_unbuild, METH_VARARGS, ""},
  {"unload",(PyCFunction)py_an_unload, METH_VARARGS, ""},
  {"get_distance",(PyCFunction)py_an_get_distance, METH_VARARGS, ""},
  {"get_n_items",(PyCFunction)py_an_get_n_items, METH_VARARGS, ""},
  {"verbose",(PyCFunction)py_an_verbose, METH_VARARGS, ""},
  {"set_seed",(PyCFunction)py_an_set_seed, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};


static PyTypeObject PyAnnoyType = {
  PyVarObject_HEAD_INIT(NULL, 0)
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

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "annoylib",          /* m_name */
    "",                  /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#endif

PyObject *create_module(void) {
  PyObject *m;

  if (PyType_Ready(&PyAnnoyType) < 0)
    return NULL;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (m == NULL)
    return NULL;

  Py_INCREF(&PyAnnoyType);
  PyModule_AddObject(m, "Annoy", (PyObject *)&PyAnnoyType);
  return m;
}

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_annoylib(void) {
    return create_module();      // it should return moudule object in py3
  }
#else
  PyMODINIT_FUNC initannoylib(void) {
    create_module();
  }
#endif


// vim: tabstop=2 shiftwidth=2
