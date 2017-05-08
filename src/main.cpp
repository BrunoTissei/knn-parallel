#include <cstdio>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <cstring>
#include <chrono>
#include <ctime>

#include "core/KnnClassifier.h"
#include "math/Metrics.h"
#include "algorithm/BallTree.h"

using namespace std;

int confusion[100][100];

void input(matrix &tr_set, matrix &ts_set, char **argv) {
  double x;
  int n_tr, k_tr;
  int n_ts, k_ts;

  FILE *training_f = fopen(argv[1], "r");
  FILE *testing_f = fopen(argv[2], "r");

  // TODO: Add verification of the input files
  fscanf(training_f, "%d %d", &n_tr, &k_tr);
  tr_set.resize(n_tr);

  for (int i = 0; i < n_tr; ++i) {
    for (int j = 0; j < k_tr; ++j) {
      fscanf(training_f, "%lf", &x);
      tr_set[i].x.push_back(x);
    }

    tr_set[i].index = i;
    fscanf(training_f, "%d", &tr_set[i].mclass);
  }

  fscanf(testing_f, "%d %d", &n_ts, &k_ts);
  ts_set.resize(n_ts);

  for (int i = 0; i < n_ts; ++i) {
    for (int j = 0; j < k_ts; ++j) {
      fscanf(testing_f, "%lf", &x);
      ts_set[i].x.push_back(x);
    }

    ts_set[i].index = i;
    fscanf(testing_f, "%d", &ts_set[i].mclass);
  }
}

template <typename function> 
double measure_time(function f) {
  chrono::duration<double> elapsed;
  chrono::time_point<chrono::system_clock> start;

  start = chrono::system_clock::now();
  f();
  elapsed = chrono::system_clock::now() - start;

  return elapsed.count() * 1000.0;
}

int main(int argc, char **argv) {
  int k = atoi(argv[3]);
  double corr;

  matrix tr_set;
  matrix ts_set;
  memset(confusion, 0, sizeof(confusion));

  input(tr_set, ts_set, argv);
  const int n_iter = ts_set.size();


  KnnClassifier<BallTree> KnnClf(Metrics::SSD(), k);

  double training_time = measure_time([&]() {
    KnnClf.fit(tr_set);
  });

  double testing_time = measure_time([&]() {
    int curr;
    corr = 0.0;

    #pragma omp parallel for reduction (+:corr) shared(KnnClf, ts_set, k, curr)
    for (int i = 0; i < n_iter; ++i) {
      int pred = KnnClf.predict(ts_set[i]);

      confusion[pred][ts_set[i].mclass]++;
      if (pred == ts_set[i].mclass)
        corr++;

      printf("%d/%d\r", curr, n_iter);

      #pragma omp atomic
      curr++;
    }
  });


  printf("%d/%d\n", n_iter, n_iter);
  printf("Done\n\n");

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

  return 0;
}
