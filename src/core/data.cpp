#include "data.h"

point *create_point(int size) {
  point *ret = new point;
  ret->x = new double[size];
  ret->size = size;

  return ret;
}
