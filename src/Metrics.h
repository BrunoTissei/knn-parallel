#ifndef _METRICS_H
#define _METRICS_H

#include <cmath>
#include <functional>

#include "BallTree.h"

class Metrics {

  public:

    static inline metric euclidean() {
      return [](const point &a, const point &b) {
        double dist = 0.0;
        for (int i = 0; i < (int) a.x.size(); ++i) {
          dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));
        }

        return sqrt(dist);
      };
    }

    static inline metric MSE() {
      return [](const point &a, const point &b) {
        double dist = 0.0;
        for (int i = 0; i < (int) a.x.size(); ++i) {
          dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));
        }

        return (dist) / a.x.size();
      };
    }

    // Sum of squared difference
    static inline metric SSD() {
      return [](const point &a, const point &b) {
        double dist = 0.0;
        for (int i = 0; i < (int) a.x.size(); ++i) {
          dist += ((a.x[i] - b.x[i]) * (a.x[i] - b.x[i]));
        }

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

#endif
