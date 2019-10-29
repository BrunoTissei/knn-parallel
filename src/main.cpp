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

struct timer {
  double *var;
  std::chrono::time_point<std::chrono::system_clock> start;

  timer(double *x) {
    start = std::chrono::system_clock::now();
    var = x;
  }

  ~timer() {
    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    *var = elapsed.count() * 1000.0;
  }
};

#define TIMER(pbn) timer measure(pbn)

bool input(vec_points &tr_set, 
    vec_points &ts_set, 
    int &k, int &nclass, 
    int argc, char **argv);

int main(int argc, char **argv) {
  int k, curr = 0;
  int nclass = -1;

  double corr = 0.0;
  double training_time, testing_time;

  vec_points tr_set, ts_set;

  if (!input(tr_set, ts_set, k, nclass, argc, argv))
    return 1;

  const int n_iter = ts_set.size();
  std::vector<std::vector<int>> confusion(nclass, std::vector<int>(nclass, 0));

  {
    KnnClassifier<BallTree> clf(Metrics::SSD(), k);

    printf("Building Tree...\n");
    {
      TIMER(&training_time);

      clf.fit(tr_set);
    }
    printf("Done\n\n");

    printf("Testing...\n");
    {
      TIMER(&testing_time);

      #pragma omp parallel for reduction (+:corr) shared(clf, ts_set, curr, nclass)
      for (int i = 0; i < n_iter; ++i) {
        int pred = clf.predict(*ts_set[i], nclass);
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

  printf("Building time: %lf ms\n", training_time);
  printf("Testing time: %lf ms\n", testing_time);

  printf("Accuracy: %lf\n\n", corr / n_iter);

  printf("Confusion matrix:\n");
  for (int i = 0; i < nclass; ++i) {
    for (int j = 0; j < nclass; ++j)
      printf("%5d ", confusion[i][j]);

    printf("\n");
  }

  for (auto i : tr_set)
    delete i;
  for (auto i : ts_set)
    delete i;

  return 0;
}

bool input(vec_points &tr_set, vec_points &ts_set, 
    int &k, int &nclass, 
    int argc, char **argv) 
{
  if (argc < 4) {
    printf("Usage:\n");
    printf("\tknn <training data> <testing data> k\n");
    return false;
  }

  FILE *training_f = fopen(argv[1], "r");
  FILE *testing_f = fopen(argv[2], "r");
  k = atoi(argv[3]);

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

  auto read_file = [&nclass](FILE *f, vec_points &s) {
    int n, d;

    fscanf(f, "%d %d", &n, &d);
    s.resize(n);

    for (int i = 0; i < n; ++i) {
      s[i] = new point;
      s[i]->x.resize(d);

      for (int j = 0; j < d; ++j)
        fscanf(f, "%lf", &(s[i]->x[j]));

      s[i]->index = i;
      fscanf(f, "%d", &s[i]->mclass);
      nclass = std::max(nclass, s[i]->mclass);
    }  
  };

  read_file(training_f, tr_set);
  read_file(testing_f, ts_set);

  fclose(training_f);
  fclose(testing_f);

  nclass++;

  return true;
}
