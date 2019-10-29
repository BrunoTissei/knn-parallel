#pragma once

#include <cmath>
#include <functional>
#include <immintrin.h>

#include "core/data.h"


typedef std::function<double(const point&, const point&)> metric; 

class Metrics {

public:
  // Euclidean Distance
  static inline metric euclidean() {
    return [](const point &a, const point &b) {
      double dist = 0.0;

      for (int i = 0; i < (int) a.x.size(); ++i) {
        dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));
      }

      return sqrt(dist);
    };
  }

  // Mean-Square Error
  static inline metric MSE() {
    return [](const point &a, const point &b) {
      double dist = 0.0;

      for (int i = 0; i < (int) a.x.size(); ++i) {
        dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));
      }

      return (dist) / a.x.size();
    };
  }

  // Sum of Squared Differences
  static inline metric SSD() {
    return [](const point &a, const point &b) {
      int i = 0, n = a.x.size() - (a.x.size() % 4);
      double dist = 0.0;

      __m256d ma, mb, msub, mans;

      for (i = 0; i < n; i += 4) {
        ma = _mm256_loadu_pd(&(a.x[i]));
        mb = _mm256_loadu_pd(&(b.x[i]));

        msub = _mm256_sub_pd(ma, mb);
        msub = _mm256_mul_pd(msub, msub);

        mans = _mm256_hadd_pd(msub, msub);
        dist += ((double*) &mans)[0] + ((double*) &mans)[2];
      }

      for (; i < (int) a.x.size(); ++i)
        dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));

      return (dist);
    };
  }

  // Sum of absolute difference
  static inline metric SAD() {
    return [](const point &a, const point &b) {
      double dist = 0.0;

      for (int i = 0; i < (int) a.x.size(); ++i) {
        dist += fabs(a.x[i] - b.x[i]);
      }

      return (dist);
    };
  }
};
