#ifndef _DATA_H
#define _DATA_H

#include <vector>

struct point {
  std::vector<double> x;
  int index, mclass;
  int size;
};

typedef std::vector<point *> matrix;

#endif
