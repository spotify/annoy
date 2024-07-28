%module annoy

namespace Annoy {}

%{
#include "annoygomodule.h"
%}


// const float *
%typemap(gotype) (const float *)  "[]float32"
%typemap(gotype) (int32_t)  "int32"

%typemap(in) (const float *)
%{
    float *v;
    vector<float> w;
    v = (float *)$input.array;
    for (int i = 0; i < $input.len; i++) {
       w.push_back(v[i]);
    }
    $1 = &w[0];
%}


%typemap(gotype) (const char *) "string"

%typemap(in) (const char *)
%{
  $1 = (char *)calloc((((_gostring_)$input).n + 1), sizeof(char));
  strncpy($1, (((_gostring_)$input).p), ((_gostring_)$input).n);
%}

%typemap(freearg) (const char *)
%{
  free($1);
%}


%ignore fill_from_vector;
%rename(X_RawAnnoyVectorInt) AnnoyVectorInt;
%rename(X_RawAnnoyVectorFloat) AnnoyVectorFloat;

%insert(go_wrapper) %{

type AnnoyVectorInt interface {
  X_RawAnnoyVectorInt
  ToSlice() []int32
  Copy(in *[]int32)
  InnerArray() []int32
  Free()
}

func NewAnnoyVectorInt() AnnoyVectorInt {
    vec := NewX_RawAnnoyVectorInt()
    return vec.(SwigcptrX_RawAnnoyVectorInt)
}

func (p SwigcptrX_RawAnnoyVectorInt) ToSlice() []int32 {
    var out []int32
    p.Copy(&out)
    return out
}

func (p SwigcptrX_RawAnnoyVectorInt) Copy(in *[]int32)  {
    out := *in
    inner := p.InnerArray()
    if cap(out) >= len(inner) {
        if len(out) != len(inner) {
          out = out[:len(inner)]
        }
    } else {
        out = make([]int32, len(inner))
    }

    copy(out, inner)
    *in = out
}

func (p SwigcptrX_RawAnnoyVectorInt) Free() {
    DeleteX_RawAnnoyVectorInt(p)
}

func (p SwigcptrX_RawAnnoyVectorInt) InnerArray() []int32 {
	length := p.Len()
    ptr := unsafe.Pointer(p.ArrayPtr())
	return ((*[1 << 30]int32)(ptr))[:length:length]
}

%}

%insert(go_wrapper) %{

type AnnoyVectorFloat interface {
  X_RawAnnoyVectorFloat
  ToSlice() []float32
  Copy(in *[]float32)
  InnerArray() []float32
  Free()
}

func NewAnnoyVectorFloat() AnnoyVectorFloat {
    vec := NewX_RawAnnoyVectorFloat()
    return vec.(SwigcptrX_RawAnnoyVectorFloat)
}

func (p SwigcptrX_RawAnnoyVectorFloat) ToSlice() []float32 {
    var out []float32
    p.Copy(&out)
    return out
}

func (p SwigcptrX_RawAnnoyVectorFloat) Copy(in *[]float32)  {
    out := *in
    inner := p.InnerArray()
    if cap(out) >= len(inner) {
        if len(out) != len(inner) {
          out = out[:len(inner)]
        }
    } else {
        out = make([]float32, len(inner))
    }

    copy(out, inner)
    *in = out
}

func (p SwigcptrX_RawAnnoyVectorFloat) Free() {
    DeleteX_RawAnnoyVectorFloat(p)
}

func (p SwigcptrX_RawAnnoyVectorFloat) InnerArray() []float32 {
    length := p.Len()
    ptr := unsafe.Pointer(p.ArrayPtr())
    return ((*[1 << 30]float32)(ptr))[:length:length]
}

%}

/* Let's just grab the original header file here */
%include "annoygomodule.h"

%feature("notabstract") GoAnnoyIndexAngular;
%feature("notabstract") GoAnnoyIndexEuclidean;
%feature("notabstract") GoAnnoyIndexManhattan;
%feature("notabstract") GoAnnoyIndexDotProduct;