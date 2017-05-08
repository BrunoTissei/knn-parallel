#include <cstdio>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <cstring>
#include <chrono>
#include <ctime>

#include "Metrics.h"
#include "BallTree.h"

using namespace std;

int confusion[100][100];

int main(int argc, char **argv) {
  FILE *training_f = fopen(argv[1], "r");
  FILE *testing_f = fopen(argv[2], "r");

  double x;
  int k = atoi(argv[3]);

  int n_tr, k_tr;
  int n_ts, k_ts;

  matrix tr_set;
  matrix ts_set;

  chrono::time_point<chrono::system_clock> start, end;
  chrono::duration<double> tr_elapsed, ts_elapsed;

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

  BallTree btree(tr_set, Metrics::SSD());

  start = chrono::system_clock::now();
  #pragma omp parallel shared(btree)
  {
    #pragma omp single
    btree.build(k);
  }
  end = chrono::system_clock::now();
  tr_elapsed = end - start;

  const int iter = ts_set.size();

  int curr = 0;
  double corr = 0;

  memset(confusion, 0, sizeof(confusion));

  start = chrono::system_clock::now();

  #pragma omp parallel for reduction (+:corr) shared(btree, ts_set, k, curr)
  for (int i = 0; i < iter; ++i) {
    matrix m;
    btree.search(ts_set[i], k, m);
    vector<int> cnt(10, 0);

    for (int j = 0; j < (int) m.size(); ++j)
      cnt[m[j].mclass]++; 

    int grt = 0, ans = 0;
    for (int j = 0; j < (int) cnt.size(); ++j) {
      if (cnt[j] > grt) {
        grt = cnt[j];
        ans = j;
      }
    }

    confusion[ans][ts_set[i].mclass]++;
    if (ans == ts_set[i].mclass)
      corr++;

    printf("%d/%d\r", curr, iter);

    #pragma omp atomic
    curr++;
  }

  end = chrono::system_clock::now();
  ts_elapsed = end - start;

  printf("%d/%d\n", iter, iter);
  printf("Done\n\n");

  printf("Training time: %lf ms\n", tr_elapsed.count() * 1000.0);
  printf("Testing time: %lf ms\n", ts_elapsed.count() * 1000.0);

  printf("Accuracy: %lf\n\n", corr / iter);

  printf("Confusion matrix:\n");
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      printf("%5d ", confusion[i][j]);
    }

    printf("\n");
  }

  return 0;
}
