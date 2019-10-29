#pragma once

#include <omp.h>
#include <vector>

#include "algorithm/ball_tree.h"
#include "core/data.h"
#include "math/metrics.h"

template <class T>
class KnnClassifier {
  
private:
  int k;
  T *tree;

public:
  /**
   * Constructor assigns metric function.
   * @param distance metric to be used
   * @param k number of neighbors
   */
  KnnClassifier(metric distance, int k);

  ~KnnClassifier();

  /**
   * Fit "training set" to classifier (i.e. builds ball-tree)
   * @param points set of points
   */
  void fit(vec_points &points);

  /**
   * Returns points's class (prediction).
   * @param point query point
   * @param nclass number of available classes
   * @return predicted class
   */
  int predict(const point &point, int nclass);
};

// Constructor assigns metric function.
template <class T>
KnnClassifier<T>::KnnClassifier(metric distance, int k) {
  this->k = k;
  this->tree = new T(distance);
}

template <class T>
KnnClassifier<T>::~KnnClassifier() {
  delete tree;
}

// Fit "training set" to classifier (i.e. builds ball-tree)
template <class T>
void KnnClassifier<T>::fit(vec_points &points) {
  #pragma omp parallel shared(tree, points)
  {
    #pragma omp single
    tree->build(points, this->k);
  }
}

// Returns points's class (prediction).
template <class T>
int KnnClassifier<T>::predict(const point &point, int nclass) {
  int result;
  int grt, pred;
  vec_points m;

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
