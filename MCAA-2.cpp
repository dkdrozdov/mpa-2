#include "pch.h"

using namespace std;

inline double *allocateMatrix(const int n) { return new double[n * n]; }
inline double *allocateVector(const int n) { return new double[n]; }
inline void deleteObject(double *o) { delete[] o; }
//void mset(double *o, int i, int j) { (& o[i])[j] = ; }

void fillRandom(double *o, const int length)
{
   for (int i = 0; i < length; i++)
      o[i] = rand() % 10;
}

void fill(double *o, int length, double filler)
{
   for (int i = 0; i < length; i++)
      o[i] = filler;
}

void  readFromFileMatrix(double *o, const int n, const string path)
{
   fstream fs;
   fs.open(path, fstream::in);

   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
         fs >> o[i * n + j];
   }
}

void writeToFileMatrix(double *o, const  int n, const string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
         fs << o[i * n + j] << "\t";
      fs << endl;
   }
}

void readFromFileVector(double *o, const int n, const string path)
{
   fstream fs;
   fs.open(path, fstream::in);

   for (int i = 0; i < n; i++)
      fs >> o[i];

   fs.close();
}

void writeToFileVector(double *o, const int n, const string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   for (int i = 0; i < n; i++)
   {
      fs << o[i] << "\t";
      fs << endl;
   }

   fs.close();
}

void readFromFileDouble(double *o, const string path)
{
   fstream fs;
   fs.open(path, fstream::in);
   fs >> *o;
   fs.close();
}

void writeToFileDouble(double *o, const string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   fs << *o << "\t";
   fs << endl;

   fs.close();
}
//task1
double scalarProduct(double *v1, double *v2, const int n)
{
   double sum = 0;
   for (int i = 0; i < n; i++)
      sum += v1[i] * v2[i];
   return sum;
}

double scalarProductParallel(double *v1, double *v2, const int n, int threads)
{
   double sum = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : sum)
   for (int i = 0; i < n; i++)
      sum += v1[i] * v2[i];
   return sum;
}

struct task1Result
{
   int vectorSize{};
   double scalarProductSerial{};
   double scalarProductParallel{};
   chrono::duration<double, std::milli> durationSerial{};
   vector<chrono::duration<double, std::milli>> durationParallels{};
   vector<double> accelerationParallels{};
};

void writeToFileTask1Results(vector<task1Result> results, const string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   int maxThreads = omp_get_max_threads();
   fs << "n" << "\t";
   fs << "ps" << "\t";
   fs << "pp" << "\t";
   fs << "ts,ms" << "\t";
   for (int i = 2; i <= maxThreads; i++)
      fs << "tp(" << i << "),ms" << "\t";
   for (int i = 2; i <= maxThreads; i++)
      fs << "a(" << i << ")" << "\t";
   fs << endl;

   for (auto &result : results)
   {
      fs << result.vectorSize << "\t";
      fs << result.scalarProductSerial << "\t";
      fs << result.scalarProductParallel << "\t";
      fs << result.durationSerial.count() << "\t";
      for (auto &durationParallel : result.durationParallels)
         fs << durationParallel.count() << "\t";
      for (auto &accelerationParallel : result.accelerationParallels)
         fs << accelerationParallel << "\t";
      fs << endl;
   }

   fs.close();
}

void task1()
{
   int maxThreads = omp_get_max_threads();
   vector<int> vectorSizes{ 10, 1000, 100000, 1000000 };
   vector<task1Result> results{};
   for (auto n : vectorSizes)
   {
      double *v1 = allocateVector(n);
      double *v2 = allocateVector(n);

      fillRandom(v1, n);
      fillRandom(v2, n);

      auto start_time = chrono::high_resolution_clock::now();
      double productSerial = scalarProduct(v1, v1, n);
      auto end_time = chrono::high_resolution_clock::now();
      chrono::duration<double, std::milli> durationSerial = end_time - start_time;

      vector<chrono::duration<double, std::milli>> durationParallels{};
      vector<double> accelerationParallels{};
      double productParallel = 0;
      for (int threads = 2; threads <= maxThreads; threads++)
      {
         start_time = chrono::high_resolution_clock::now();
         productParallel = scalarProductParallel(v1, v1, n, threads);
         auto end_time = chrono::high_resolution_clock::now();
         chrono::duration<double, std::milli> durationParallel = end_time - start_time;

         durationParallels.push_back(durationParallel);
         accelerationParallels.push_back(durationSerial / durationParallel);
      }

      task1Result result{};
      result.vectorSize = n;
      result.scalarProductSerial = productSerial;
      result.scalarProductParallel = productParallel;
      result.durationSerial = durationSerial;
      result.durationParallels = durationParallels;
      result.accelerationParallels = accelerationParallels;

      results.push_back(result);
      deleteObject(v1);
      deleteObject(v2);
   }

   writeToFileTask1Results(results, "resultsTask1.txt");
}
//task2
double *matrixMultiplication(double *m1, double *m2, int n)
{
   double *mresult = allocateMatrix(n);

   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         double sum = 0;
         for (int k = 0; k < n; k++)
         {
            sum += m1[i * n + k] * m2[k * n + j];
         }
         mresult[i * n + j] = sum;
      }
   }
   return mresult;
}

double *matrixMultiplicationParallel(double *m1, double *m2, int n, int threads)
{
   double *mresult = allocateMatrix(n);

#pragma omp parallel for num_threads(threads)
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         double sum = 0;
         for (int k = 0; k < n; k++)
         {
            sum += m1[i * n + k] * m2[k * n + j];
         }
#pragma omp critical
         {
            mresult[i * n + j] = sum;
         }
      }
   }
   return mresult;
}

double matrixNorm(double *m, int n)
{
   double sum = 0;
   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         sum += m[i * n + j] * m[i * n + j];
   return sqrt(sum);
}

struct task2Result
{
   chrono::duration<double, std::milli> duration{};
   double acceleration;
   double norm;
};

void writeToFileTask2Results(vector<task2Result> results, string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   fs << "threads" << "\t" << "t,ms" << "\t" << "a" << "\t" << "norm" << endl;
   int maxThreads = omp_get_max_threads();
   for (int i = 0; i < results.size(); i++)
   {
      fs << i + 1 << "\t";
      fs << results[i].duration.count() << "\t";
      fs << results[i].acceleration << "\t";
      fs << results[i].norm;
      fs << endl;
   }

   fs.close();
}

void task2()
{
   int maxThreads = omp_get_max_threads();
   int n = 250;
   vector<task2Result> results;

   for (int threads = 1; threads <= maxThreads; threads++)
   {
      double *m1 = allocateMatrix(n);
      double *m2 = allocateMatrix(n);

      fillRandom(m1, n * n);
      fillRandom(m2, n * n);

      auto start_time = chrono::high_resolution_clock::now();
      double *product = threads == 1 ? matrixMultiplication(m1, m2, n) : matrixMultiplicationParallel(m1, m2, n, threads);
      auto end_time = chrono::high_resolution_clock::now();
      chrono::duration<double, std::milli> duration = end_time - start_time;


      task2Result result{};
      result.acceleration = threads == 1 ? 0 : results[0].duration / duration;
      result.duration = duration;
      result.norm = matrixNorm(product, n);

      results.push_back(result);
      deleteObject(m1);
      deleteObject(m2);
      deleteObject(product);
   }

   writeToFileTask2Results(results, "resultsTask2.txt");
}

void genU(double *m, int n)
{
   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         m[i * n + j] = (j < i) ? 0 : (i == j) ? rand() % 10 + 10 : rand() % 5 + 1;
}

double *slae(double *u, double *b, int n)
{
   double *x = allocateVector(n);

   for (int i = n - 1; i >= 0; i--)
   {
      double sum = 0;
      for (int j = i + 1; j < n; j++)
         sum += u[i * n + j] * x[j];
      x[i] = (b[i] - sum) / u[i * n + i];
   }

   return x;
}

double *slaeParallel(double *u, double *b, int n, int threads)
{
   double *x = allocateVector(n);

   for (int i = n - 1; i >= 0; i--)
   {
      double sum = 0;
#pragma omp parallel for num_threads(threads) reduction(+:sum)
      for (int j = i + 1; j < n; j++)
         sum += u[i * n + j] * x[j];
      x[i] = (b[i] - sum) / u[i * n + i];
   }

   return x;
}

double *matrixVectorMultiplication(double *m, double *v, int n)
{
   double *vresult = allocateVector(n);

   for (int i = 0; i < n; i++)
   {
      vresult[i] = 0;
      for (int j = 0; j < n; j++)
         vresult[i] += m[i * n + j] * v[j];
   }
   return vresult;
}

struct task3Result
{
   chrono::duration<double, std::milli> duration;
   double acceleration;
   double normOriginal;
   double normCalculated;
};

void writeToFileTask3Results(vector<task3Result> results, string path)
{
   fstream fs;
   fs.open(path, fstream::out);

   fs << "threads" << "\t" << "t,ms" << "\t" << "a" << "\t" << "norm original" << "\t" << "norm calculated" << endl;
   int maxThreads = omp_get_max_threads();
   for (int i = 0; i < results.size(); i++)
   {
      fs << i + 1 << "\t";
      fs << results[i].duration.count() << "\t";
      fs << results[i].acceleration << "\t";
      fs << results[i].normOriginal << "\t";
      fs << results[i].normCalculated;
      fs << endl;
   }

   fs.close();
}

double vectorNorm(double *m, int n)
{
   double sum = 0;
   for (int i = 0; i < n; i++)
      sum += m[i] * m[i];
   return sqrt(sum);
}

void task3()
{
   int maxThreads = omp_get_max_threads();
   int n = 2500;
   vector<task3Result> results;

   for (int threads = 1; threads <= maxThreads; threads++)
   {
      double *u = allocateMatrix(n);
      double *x = allocateVector(n);
      genU(u, n);
      fillRandom(x, n);
      double *b = matrixVectorMultiplication(u, x, n);

      auto start_time = chrono::high_resolution_clock::now();
      double *xCalculated = threads == 1 ? slae(u, b, n) : slaeParallel(u, b, n, maxThreads);
      auto end_time = chrono::high_resolution_clock::now();
      chrono::duration<double, std::milli> duration = end_time - start_time;

      task3Result result{};
      result.acceleration = threads == 1 ? 0 : results[0].duration / duration;
      result.duration = duration;
      result.normOriginal = vectorNorm(x, n);
      result.normCalculated = vectorNorm(xCalculated, n);

      results.push_back(result);
      deleteObject(u);
      deleteObject(x);
      deleteObject(b);
      deleteObject(xCalculated);
   }

   writeToFileTask3Results(results, "resultsTask3.txt");
}

int main()
{
   srand(time(0));
   //   task1();
   //   task2();
   task3();
}