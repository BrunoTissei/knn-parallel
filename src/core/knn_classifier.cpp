#include "core/knn_classifier.h"

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
  #pragma omp parallel shared(tree,points)
  {
    #pragma omp single
    tree->build(&points, this->k);
  }
}

template <class T>
int KnnClassifier<T>::predict(const point &point) {
  matrix m;
  int result;
  int grt, pred;

  std::vector<int> cnt(50,0);

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


template class KnnClassifier<BallTree>;
