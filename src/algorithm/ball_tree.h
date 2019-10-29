#pragma once

#include <set>
#include <omp.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <functional>

#include "core/data.h"
#include "math/metrics.h"

using prio_queue = std::multiset<std::pair<double,int>>;

struct node {
  int index;
  bool leaf;
  double radius;

  point center;
  std::vector<int> points;

  node *left, *right;
};

class BallTree {

private:
  int k;
  node *root;
  vec_points data;
  metric distance;

public:
  /**
   * Constructor assigns metric function.
   * @param distance metric to be used
   */
  BallTree(metric distance);

  ~BallTree();

  /**
   * Builds ball-tree from points.
   * @param points vector of points
   * @param k the value of k from [K]NN neighbors
   */
  void build(vec_points &points, int k);

  /**
   * Searches for k nearest neighbors of point t.
   * @param t query point
   * @param k number of neighbors
   * @param ans output (vector of points)
   */
  void search(const point &t, int k, vec_points &ans);

private:
  /**
   * Searches recursively.
   * @param n root of tree
   * @param t query point
   * @param pq priority queue used during search
   * @param k number of neighbors
   */
  void search(node *n, const point &t, prio_queue &pq, int k);

  /**
   * Destroys tree recursively.
   * @param n root of tree
   */
  void clear(node *n);

  /**
   * Returns approximate center of set of points.
   * @param points vector of points
   * @param center resulting point
   */
  void get_center(vec_points &points, point &center);

  /**
   * Creates partitions (left and right from center) of points.
   * @param points set of points
   * @param left, right resulting partitions
   * @param left_ind index of left-most point
   */
  void partition(vec_points &points, 
      vec_points &left, 
      vec_points &right, 
      int left_ind);

  /**
   * Returns radius of sphere.
   * @param center center of sphere
   * @param points set of points
   * @return pair(radius, left-most point index)
   */
  std::pair<double,int> get_radius(point &center, 
      vec_points &points);

  /**
   * Builds tree recursively.
   * @param points set of points
   * @return root of tree
   */
  node *build(vec_points &points);
};
