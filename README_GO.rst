Install
-------

To install, you'll need Swig (tested with Swig 4.2.1 on Ubuntu 24.04), and then just::

  swig -go -intgosize 64 -cgo -c++ src/annoygomodule.i
  mkdir -p $(go env GOPATH)/src/annoy
  cp src/annoygomodule_wrap.cxx src/annoy.go src/annoygomodule.h src/annoylib.h src/kissrandom.h test/annoy_test.go $(go env GOPATH)/src/annoy
  cd $(go env GOPATH)/src/annoy
  go mod init github.com/spotify/annoy
  go mod tidy
  go test

Background
----------

See the main README.

Go code example
-------------------

.. code-block:: go

  package main
  
  import (
         "fmt"
         "math/rand"

         "github.com/spotify/annoy"
  )
  
  func main() {
       f := 40
       t := annoy.NewAnnoyIndexAngular(f)
       for i := 0; i < 1000; i++ {
       	 item := make([]float32, 0, f)
       	 for x:= 0; x < f; x++ {
  	     item = append(item, rand.Float32())
  	 }
  	 t.AddItem(i, item)
       }
       t.Build(10)
       t.Save("test.ann")
  
       annoy.DeleteAnnoyIndexAngular(t)
       
       t = annoy.NewAnnoyIndexAngular(f)
       t.Load("test.ann")
       
       result := annoyindex.NewAnnoyVectorInt()
       defer result.Free()
       t.GetNnsByItem(0, 1000, -1, result)
       fmt.Printf("%v\n", result.ToSlice())
  
  }
  
Right now it only accepts integers as identifiers for items. Note that it will allocate memory for max(id)+1 items because it assumes your items are numbered 0 … n-1. If you need other id's, you will have to keep track of a map yourself.

Full Go API
---------------

See annoygomodule.h. Generally the same as Python API except some arguments are not optional. Go binding does not support multithreaded build.

Tests
-------
A simple test is supplied in test/annoy_test.go.

Discuss
-------

Memroy leak in the previous versions has been fixed thanks to https://github.com/swig/swig/issues/2292. (memory leak fix is implemented in https://github.com/Rikanishu/annoy-go)

Go glue written by Taneli Leppä (@rosmo). You can contact me via email (see https://github.com/rosmo).
