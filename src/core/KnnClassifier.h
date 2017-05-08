#ifndef _KNN_CLASSIFIER_H
#define _KNN_CLASSIFIER_H

#include <omp.h>

#include "algorithm/BallTree.h"
#include "core/data.h"
#include "math/Metrics.h"

template <class T>
class KnnClassifier {
  
  int k;
  T *tree;

  public:

    KnnClassifier(metric distance, int k);

    ~KnnClassifier();

    void fit(matrix &points);

    int predict(point &point);

};

#endif
