#ifndef _DATA_H
#define _DATA_H

#include <vector>

typedef struct point {
  std::vector<double> x;
  int index, mclass;
} point;

typedef std::vector<point> matrix;

#endif