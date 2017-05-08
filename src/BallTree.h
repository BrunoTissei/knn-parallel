#ifndef _BALL_TREE_H
#define _BALL_TREE_H

#include <set>
#include <omp.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <functional>

typedef struct point {
  std::vector<double> x;
  int index, mclass;
} point;

typedef std::vector<point> matrix;
typedef std::multiset<std::pair<double,int>> prio_queue;
typedef std::function<double(const point&, const point&)> metric; 

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
  metric distance;

  public:
    matrix data;

    BallTree(matrix &points, metric distance);
    ~BallTree();

    node *build(matrix &points);

    void build(int k);
    void partition(matrix &points, matrix &left, matrix &right, int left_ind);
    void get_center(matrix &points, point &center);
    void search(point t, int k, matrix &ans);

    std::pair<double,int> get_radius(point &center, matrix &points);

  private:
    void search(node *n, point &t, prio_queue &pq, int k);
    void clear(node *n);

};

#endif
