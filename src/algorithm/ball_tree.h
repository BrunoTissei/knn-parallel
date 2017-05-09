#ifndef _BALL_TREE_H
#define _BALL_TREE_H

#include <set>
#include <omp.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <functional>

#include "core/data.h"
#include "math/metrics.h"

typedef std::multiset<std::pair<double,int>> prio_queue;

typedef struct node {
  int index;
  bool leaf;
  double radius;

  point center;
  std::vector<int> points;

  node *left, *right;
} node;

class BallTree {

  int k;
  node *root;
  matrix data;
  metric distance;

  public:

    BallTree(metric distance);

    ~BallTree();

    void build(matrix &points, int k);

    void search(const point &t, int k, matrix &ans);

  private:

    void search(node *n, const point &t, prio_queue &pq, int k);

    void clear(node *n);

    void get_center(matrix &points, point &center);

    void partition(matrix &points, matrix &left, matrix &right, int left_ind);

    std::pair<double,int> get_radius(point &center, matrix &points);

    node *build(matrix &points);

};

#endif
