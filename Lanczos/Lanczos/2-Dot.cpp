//#include <iostream>
//#include <mpi.h>
//#include <omp.h>
//#include "mpi_dot.h"
//using namespace std;
//int main(int argc, char **argv)
//{
//	MPI_Init(&argc, &argv);
//
//	int rank, size;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//	MPI_Comm comm = MPI_COMM_WORLD;
//
//	// Initialize two vectors of the same length and then compute the dot product
//	int N = 10;
//	float* v1;
//	float* v2;
//	v1 = new float[N];
//	v2 = new float[N];
//	for (int i = 0; i < N; i++) { v1[i] = (float)(i+1); v2[i] = (float)(i+1); }
//
//	float final_sum = 0.0f;
//
//	mpi_dot(v1, v2, &final_sum, N / size, rank, size, comm);
//
//	cout << "v1 * v2 = " << sqrt(final_sum) << endl;
//
//	MPI_Finalize();
//
//	return 0;
//}