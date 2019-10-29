#pragma once

#include <omp.h>
#include <vector>

#include "algorithm/ball_tree.h"
#include "core/data.h"
#include "math/metrics.h"

template <class T>
class KnnClassifier {
  
  int k;
  T *tree;

  public:
    KnnClassifier(metric distance, int k);

    ~KnnClassifier();

    void fit(matrix &points);

    int predict(const point &point, int nclass);
};

template <class T>
KnnClassifier<T>::KnnClassifier(metric distance, int k) {
  this->k = k;
  this->tree = new T(distance);
}

template <class T>
KnnClassifier<T>::~KnnClassifier() {
  delete tree;
}

template <class T>
void KnnClassifier<T>::fit(matrix &points) {
  #pragma omp parallel shared(tree, points)
  {
    #pragma omp single
    tree->build(points, this->k);
  }
}

template <class T>
int KnnClassifier<T>::predict(const point &point, int nclass) {
  int result;
  int grt, pred;
  matrix m;

  std::vector<int> cnt(nclass, 0);
  tree->search(point, k, m);

  for (int j = 0; j < (int) m.size(); ++j) {
    result = m[j]->mclass;
    cnt[result]++; 
  }

  grt = pred = 0;
  for (int j = 0; j < (int) cnt.size(); ++j) {
    if (cnt[j] > grt) {
      grt = cnt[j];
      pred = j;
    }
  }

  return pred;
}
