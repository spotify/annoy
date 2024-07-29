#include "annoylib.h"
#include "kissrandom.h"

using namespace Annoy;

namespace GoAnnoy {


class AnnoyVectorFloat {
    protected:
        float *ptr;
        int len;

    public:
      ~AnnoyVectorFloat() {
        free(ptr);
      };
      float* ArrayPtr() {
        return ptr;
      };
      int Len() {
        return len;
      };
      float Get(int i) {
        if (i >= len) {
            return 0.0;
        }
        return ptr[i];
      };
      void fill_from_vector(vector<float>* v) {
            if (ptr != NULL) {
               free(ptr);
            }
            ptr = (float*) malloc(v->size() * sizeof(float));
            for (int i = 0; i < v->size(); i++) {
                ptr[i] = (float)(*v)[i];
            }
            len = v->size();
      };
};

class AnnoyVectorInt {
    protected:
        int32_t *ptr;
        int len;

    public:
      ~AnnoyVectorInt() {
        free(ptr);
      };
      int32_t* ArrayPtr() {
        return ptr;
      };
      int Len() {
        return len;
      };
      int32_t Get(int i) {
        if (i >= len) {
            return 0.0;
        }
        return ptr[i];
      };
      void fill_from_vector(vector<int32_t>* v) {
            if (ptr != NULL) {
                free(ptr);
            }
            ptr = (int32_t*) malloc(v->size() * sizeof(int32_t));
            for (int i = 0; i < v->size(); i++) {
                ptr[i] = (int32_t)(*v)[i];
            }
            len = v->size();
      };
};

class AnnoyIndex {
 protected:
  ::AnnoyIndexInterface<int32_t, float> *ptr;

  int f;

 public:
  ~AnnoyIndex() {
    delete ptr;
  };
  void addItem(int item, const float* w) {
    ptr->add_item(item, w);
  };
  void build(int q) {
    ptr->build(q, 1);
  };
  bool save(const char* filename, bool prefault) {
    return ptr->save(filename, prefault);
  };
  bool save(const char* filename) {
    return ptr->save(filename, true);
  };
  void unload() {
    ptr->unload();
  };
  bool load(const char* filename, bool prefault) {
    return ptr->load(filename, prefault);
  };
  bool load(const char* filename) {
    return ptr->load(filename, true);
  };
  float getDistance(int i, int j) {
    return ptr->get_distance(i, j);
  };
  void getNnsByItem(int item, int n, int search_k, AnnoyVectorInt* out_result, AnnoyVectorFloat* out_distances) {
    vector<int32_t>* result = new vector<int32_t>();
    vector<float>* distances = new vector<float>();

    ptr->get_nns_by_item(item, n, search_k, result, distances);

    out_result->fill_from_vector(result);
    out_distances->fill_from_vector(distances);
    delete result;
    delete distances;
  };
  void getNnsByVector(const float* w, int n, int search_k, AnnoyVectorInt* out_result, AnnoyVectorFloat* out_distances) {
    vector<int32_t>* result = new vector<int32_t>();
    vector<float>* distances = new vector<float>();

    ptr->get_nns_by_vector(w, n, search_k, result, distances);

    out_result->fill_from_vector(result);
    out_distances->fill_from_vector(distances);
    delete result;
    delete distances;
  };
  void getNnsByItem(int item, int n, int search_k, AnnoyVectorInt* out_result) {
    vector<int32_t>* result = new vector<int32_t>();

    ptr->get_nns_by_item(item, n, search_k, result, NULL);

    out_result->fill_from_vector(result);
    delete result;
  };
  void getNnsByVector(const float* w, int n, int search_k, AnnoyVectorInt* out_result) {
    vector<int32_t>* result = new vector<int32_t>();

    ptr->get_nns_by_vector(w, n, search_k, result, NULL);

    out_result->fill_from_vector(result);
    delete result;
  };

  int getNItems() {
    return (int)ptr->get_n_items();
  };
  void verbose(bool v) {
    ptr->verbose(v);
  };
  void getItem(int item, AnnoyVectorFloat *v) {
    vector<float>* r = new vector<float>();
    r->resize(this->f);
    ptr->get_item(item, &r->front());
    v->fill_from_vector(r);
  };
  bool onDiskBuild(const char* filename) {
    return ptr->on_disk_build(filename);
  };
};

class AnnoyIndexAngular : public AnnoyIndex 
{
 public:
  AnnoyIndexAngular(int f) {
    ptr = new ::AnnoyIndex<int32_t, float, ::Angular, ::Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
    this->f = f;
  }
};

class AnnoyIndexEuclidean : public AnnoyIndex {
 public:
  AnnoyIndexEuclidean(int f) {
    ptr = new ::AnnoyIndex<int32_t, float, ::Euclidean, ::Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
    this->f = f;
  }
};

class AnnoyIndexManhattan : public AnnoyIndex {
 public:
  AnnoyIndexManhattan(int f) {
    ptr = new ::AnnoyIndex<int32_t, float, ::Manhattan, ::Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
    this->f = f;
  }
};

class AnnoyIndexDotProduct : public AnnoyIndex {
 public:
  AnnoyIndexDotProduct(int f) {
    ptr = new ::AnnoyIndex<int32_t, float, ::DotProduct, ::Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
    this->f = f;
  }
};
}
