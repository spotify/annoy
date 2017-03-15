Install
-------

To install, you'll need Swig (tested with Swig 3.0.6 on OS X), and then just::

  swig -go -intgosize 64 -cgo -c++ src/annoygomodule.i
  mkdir -p $GOPATH/src/annoyindex
  cp src/annoygomodule_wrap.cxx src/annoyindex.go src/annoygomodule.h src/annoylib.h src/kissrandom.h test/annoy_test.go $GOPATH/src/annoyindex
  cd $GOPATH/src/annoyindex
  go get -t ...
  go test
  go build

Background
----------

See the main README.

Go code example
-------------------

.. code-block:: go

  package main
  
  import (
         "annoyindex"
         "fmt"
         "math/rand"
  )
  
  func main() {
       f := 40
       t := annoyindex.NewAnnoyIndexAngular(f)
       for i := 0; i < 1000; i++ {
       	 item := make([]float32, 0, f)
       	 for x:= 0; x < f; x++ {
  	     item = append(item, rand.Float32())
  	 }
  	 t.AddItem(i, item)
       }
       t.Build(10)
       t.Save("test.ann")
  
       annoyindex.DeleteAnnoyIndexAngular(t)
       
       t = annoyindex.NewAnnoyIndexAngular(f)
       t.Load("test.ann")
       
       var result []int
       t.GetNnsByItem(0, 1000, -1, &result)
       fmt.Printf("%v\n", result)
  
  }
  
Right now it only accepts integers as identifiers for items. Note that it will allocate memory for max(id)+1 items because it assumes your items are numbered 0 … n-1. If you need other id's, you will have to keep track of a map yourself.

Full Go API
---------------

See annoygomodule.h. Generally the same as Python API except some arguments are not optional. 

Tests
-------
A simple test is supplied in test/annoy_test.go.

Discuss
-------

There might be some memory leaks.

Go glue written by Taneli Leppä (@rosmo). You can contact me via email (see https://github.com/rosmo).
