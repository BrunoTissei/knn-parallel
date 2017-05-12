#include <cstdio>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <cstring>
#include <chrono>
#include <ctime>

#include "math/metrics.h"
#include "core/knn_classifier.h"
#include "algorithm/ball_tree.h"

using namespace std;

int confusion[100][100];

struct timer {
  double *var;
  chrono::time_point<chrono::system_clock> start;

  timer(double *x) {
    start = chrono::system_clock::now();
    var = x;
  }

  ~timer() {
    chrono::duration<double> elapsed = chrono::system_clock::now() - start;
    *var = elapsed.count() * 1000.0;
  }
};

#define TIMER(pbn) timer measure(pbn)

bool input(matrix &tr_set, matrix &ts_set, int &k, int argc, char **argv);

int main(int argc, char **argv) {
  int k, curr = 0;

  double corr = 0.0;
  double training_time, testing_time;

  matrix tr_set, ts_set;

  memset(confusion, 0, sizeof(confusion));
  if (!input(tr_set, ts_set, k, argc, argv)) {
    return 1;
  }

  const int n_iter = ts_set.size();

  {
    KnnClassifier<BallTree> KnnClf(Metrics::SSD(), k);

    printf("Training...\n");
    {
      TIMER(&training_time);

      KnnClf.fit(tr_set);
    }
    printf("Done\n\n");

    printf("Testing...\n");
    {
      TIMER(&testing_time);

      #pragma omp parallel for reduction (+:corr) shared(KnnClf, ts_set, k, curr)
      for (int i = 0; i < n_iter; ++i) {
        int pred = KnnClf.predict(*ts_set[i]);

        confusion[pred][ts_set[i]->mclass]++;
        if (pred == ts_set[i]->mclass)
          corr++;

        printf("%d/%d\r", curr, n_iter);

        #pragma omp atomic
        curr++;
      }

    }
    printf("%d/%d\n", n_iter, n_iter);
    printf("Done\n\n");
  }

  printf("Training time: %lf ms\n", training_time);
  printf("Testing time: %lf ms\n", testing_time);

  printf("Accuracy: %lf\n\n", corr / n_iter);

  printf("Confusion matrix:\n");
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      printf("%5d ", confusion[i][j]);
    }

    printf("\n");
  }

  for (auto i : tr_set)
    delete i;
  for (auto i : ts_set)
    delete i;

  return 0;
}

bool input(matrix &tr_set, matrix &ts_set, int &k, int argc, char **argv) {
  double x;
  int n_tr, k_tr;
  int n_ts, k_ts;

  if (argc < 4) {
    printf("Usage:\n");
    printf("\tknn <training data> <testing data> k\n");
    return false;
  }

  FILE *training_f = fopen(argv[1], "r");
  FILE *testing_f = fopen(argv[2], "r");

  if (training_f == NULL) {
    printf("ERROR:\n");
    printf("\tFile %s does not exist!\n", argv[1]);
    return false;
  }

  if (testing_f == NULL) {
    printf("ERROR:\n");
    printf("\tFile %s does not exist!\n", argv[2]);
    return false;
  }

  fscanf(training_f, "%d %d", &n_tr, &k_tr);
  tr_set.resize(n_tr);

  for (int i = 0; i < n_tr; ++i) {
    tr_set[i] = create_point(k_tr);

    for (int j = 0; j < k_tr; ++j) {
      fscanf(training_f, "%lf", &x);
      tr_set[i]->x[j] = x;
    }

    tr_set[i]->index = i;
    fscanf(training_f, "%d", &tr_set[i]->mclass);
  }

  fscanf(testing_f, "%d %d", &n_ts, &k_ts);
  ts_set.resize(n_ts);

  for (int i = 0; i < n_ts; ++i) {
    ts_set[i] = create_point(k_ts);

    for (int j = 0; j < k_ts; ++j) {
      fscanf(testing_f, "%lf", &x);
      ts_set[i]->x[j] = x;
    }

    ts_set[i]->index = i;
    fscanf(testing_f, "%d", &ts_set[i]->mclass);
  }

  k = atoi(argv[3]);

  return true;
}
