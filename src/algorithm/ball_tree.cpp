#include "algorithm/ball_tree.h"

BallTree::BallTree(metric distance) {
  this->distance = distance;
}

BallTree::~BallTree() {
  data->clear();
  clear(root);
}

void BallTree::clear(node *n) {
  if (n == nullptr) {
    return;
  }

  clear(n->left);
  clear(n->right);
  delete n;
}

void BallTree::build(matrix *points, int k) {
  this->k = k;
  this->data = points;
  this->root = build(points);
}

node *BallTree::build(matrix *points) {
  if ((*points).size() == 0) {
    return nullptr;
  }

  node *n = new node;
  n->center = get_center(points);

  if ((int) points->size() <= this->k) {
    n->points = points;

    n->radius = 0.0;
    n->leaf = true;

    n->left = nullptr;
    n->right = nullptr;
  } else {
    auto result = get_radius(n->center, points);

    matrix *l_partition = new matrix;
    matrix *r_partition = new matrix;
    partition(points, l_partition, r_partition, result.second);

    n->radius = result.first;
    n->leaf = false;

    #pragma omp task
    n->left = build(l_partition);
    #pragma omp task
    n->right = build(r_partition);
    #pragma omp taskwait
  }

  return n;
}

void BallTree::search(const point &t, int k, matrix &ans) {
  prio_queue pq;
  search(root, &t, pq, k);
  ans.clear();

  for (auto i : pq) {
    ans.push_back((*data)[i.second]);
  }
}

void BallTree::search(node *n, const point *t, prio_queue &pq, int k) {
  if (n == nullptr) {
    return;
  }

  auto top = [&](void) {
    return pq.rbegin();
  };

  if (n->leaf) {
    for (int i = 0; i < (int) n->points->size(); ++i) {
      double dist = distance(t, n->points[0][i]);

      if (pq.size() == 0 || dist < distance(t, (*data)[top()->second])) {
        pq.insert(std::make_pair(dist, n->points[0][i]->index));
        if ((int) pq.size() > k) {
          pq.erase(std::prev(pq.end()));
        }
      }
    }
  } else {
    bool cond = (pq.size() == 0);
    double dist_left = distance(t, n->left->center);
    double dist_right = distance(t, n->right->center);

    if (dist_left <= dist_right) {
      if (cond || (dist_left <= (top()->first + n->left->radius))) {
        search(n->left, t, pq, k);
      }

      if (cond || (dist_right <= (top()->first + n->right->radius))) {
        search(n->right, t, pq, k);
      }
    } else {
      if (cond || (dist_right <= (top()->first + n->right->radius))) {
        search(n->right, t, pq, k);
      }

      if (cond || (dist_left <= (top()->first + n->left->radius))) {
        search(n->left, t, pq, k);
      }
    }
  }
}

void BallTree::partition(matrix *points, matrix *left, matrix *right, 
    int left_ind) {

  int right_ind = 0;
  double dist, grt_dist = 0.0;
  double left_dist, right_dist;

  point *rm_point;
  point *lm_point = points[0][left_ind];

  for (int i = 0; i < (int) points->size(); ++i) {
    dist = distance(lm_point, points[0][i]);

    if (dist > grt_dist) {
      grt_dist = dist;
      right_ind = i;
    }
  }

  rm_point = points[0][right_ind];

  for (int i = 0; i < (int) points->size(); ++i) {
    left_dist = distance(points[0][i], lm_point);
    right_dist = distance(points[0][i], rm_point);

    if (left_dist < right_dist) {
      left->push_back(points[0][i]);
    } else {
      right->push_back(points[0][i]);
    }
  }
}

std::pair<double,int> BallTree::get_radius(point *center, const matrix *points) {
  int index = 0;
  double dist, radius = 0.0;

  for (int i = 0; i < (int) points->size(); ++i) {
    dist = distance(center, points[0][i]);

    if (radius < dist) {
      radius = dist;
      index = i;
    }
  }
  
  return std::make_pair(radius, index);
}

point *BallTree::get_center(const matrix *points) {
  point *center = new point;
  center->x.resize(points[0][0]->x.size());

  for (auto p : *points) {
    int i = 0;
    for (double dim : p->x) {
      center->x[i++] += dim;
    }
  }

  double div = 1.0 / ((double) points[0].size());
  for (int i = 0; i < (int) points[0][0]->x.size(); ++i) {
    center->x[i] *= div;
  }

  return center;
}
