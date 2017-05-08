#include "BallTree.h"

BallTree::BallTree(metric distance) {
  this->distance = distance;
}

BallTree::~BallTree() {
  data.clear();
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

void BallTree::build(matrix &points, int k) {
  this->k = k;
  this->data = points;
  this->root = build(this->data);
}

node *BallTree::build(matrix &points) {
  node *n = new node;
  get_center(points, n->center);

  if ((int) points.size() <= this->k) {
    for (int i = 0; i < (int) points.size(); ++i) {
      n->points.push_back(points[i].index);
    }

    n->leaf = true;
    n->left = nullptr;
    n->right = nullptr;
  } else {
    auto result = get_radius(n->center, points);
    n->radius = result.first;

    matrix l_partition, r_partition;
    partition(points, l_partition, r_partition, result.second);

    n->leaf = false;
    #pragma omp task
    n->left = build(l_partition);
    #pragma omp task
    n->right = build(r_partition);
    #pragma omp taskwait
  }

  return n;
}

// TODO: return vector of mclasses instead of points
void BallTree::search(point t, int k, matrix &ans) {
  prio_queue pq;
  search(root, t, pq, k);
  ans.clear();

  for (auto i : pq) {
    ans.push_back(data[i.second]);
  }
}

void BallTree::search(node *n, point &t, prio_queue &pq, int k) {
  if (n == nullptr) {
    return;
  }

  auto top = [&](void) {
    return data[pq.rbegin()->second];
  };

  if (n->leaf) {
    for (int i = 0; i < (int) n->points.size(); ++i) {
      double dist = distance(t, data[n->points[i]]);

      if (pq.size() == 0 || dist < distance(t, top())) {
        pq.insert(std::make_pair(dist, n->points[i]));
        if ((int) pq.size() > k) {
          pq.erase(std::prev(pq.end()));
        }
      }
    }
  } else {
    bool cond = (pq.size() == 0) || n->leaf;
    double dist_left = distance(t, n->left->center);
    double dist_right = distance(t, n->right->center);

    if (dist_left <= dist_right) {
      if (cond || (dist_left <= pq.rbegin()->first + n->left->radius)) {
        search(n->left, t, pq, k);
      }

      if (cond || (dist_right <= pq.rbegin()->first + n->right->radius)) {
        search(n->right, t, pq, k);
      }
    } else {
      if (cond || (dist_right <= pq.rbegin()->first + n->right->radius)) {
        search(n->right, t, pq, k);
      }

      if (cond || (dist_left <= pq.rbegin()->first + n->left->radius)) {
        search(n->left, t, pq, k);
      }
    }
  }
}

void BallTree::partition(matrix &points, matrix &left, matrix &right, 
    int left_ind) {

  int right_ind = 0;
  double dist, grt_dist = 0.0;
  double left_dist, right_dist;

  point rm_point;
  point lm_point = points[left_ind];

  for (int i = 0; i < (int) points.size(); ++i) {
    dist = distance(lm_point, points[i]);

    if (dist > grt_dist) {
      grt_dist = dist;
      right_ind = i;
    }
  }

  rm_point = points[right_ind];

  for (int i = 0; i < (int) points.size(); ++i) {
    left_dist = distance(points[i], lm_point);
    right_dist = distance(points[i], rm_point);

    if (left_dist < right_dist) {
      left.push_back(points[i]);
    } else {
      right.push_back(points[i]);
    }
  }
}

std::pair<double,int> BallTree::get_radius(point &center, matrix &points) {
  int index = 0;
  double dist, radius = 0.0;

  for (int i = 0; i < (int) points.size(); ++i) {
    dist = distance(center, points[i]);
    if (radius < dist) {
      radius = dist;
      index = i;
    }
  }
  
  return std::make_pair(radius, index);
}

void BallTree::get_center(matrix &points, point &center) {
  center.x.resize(points[0].x.size());

  for (auto p : points) {
    int i = 0;
    for (double dim : p.x) {
      center.x[i++] += dim;
    }
  }

  double div = 1.0 / ((double) points.size());
  for (int i = 0; i < (int) points[0].x.size(); ++i) {
    center.x[i] *= div;
  }
}
