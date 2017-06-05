#include <iostream>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include "mpi_dot.h"
#include "CubeFD.h"
#include "mkl.h"
using namespace std;
int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	srand(time(NULL));

	// Declarations
	const int d = 30;
	const int N = d*d*d;
	const int K = 30;
	float** R;
	float** Q;
	float* alpha;
	float* beta;
	float dot_sum;

	// Allocations
	R = new float*[K + 1];
	Q = new float*[K + 1];
	for (int i = 0; i < K + 1; i++)
	{
		R[i] = new float[N]();
		Q[i] = new float[N]();
	}
	alpha = new float[K + 1]();
	beta = new float[K + 2]();

	// Initializations
	for (int i = 0; i < N; i++)
		R[0][i] = ((float)rand()) / ((float)RAND_MAX);
	dot_sum = 0.0f;
	mpi_dot(R[0], R[0], &dot_sum, N, rank, size, MPI_COMM_WORLD);
	float* temp_r;
	temp_r = new float[N];
	mpi_times(R[0], temp_r, 1.0f / sqrt(dot_sum), N, rank, size, MPI_COMM_WORLD);
	beta[0] = sqrt(dot_sum);

	// Run Experiment
	float* temp_v1; // A*q_j
	float* temp_v2; // alpha_j * q_j
	float* temp_v3; // beta_j * q_(j-1)
	float* temp_v4; // temp_v1 - temp_v2
	temp_v1 = new float[N];
	temp_v2 = new float[N];
	temp_v3 = new float[N];
	temp_v4 = new float[N];
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);

	for (int j = 1; j < K; j++)
	{
		// q_j = r_(j-1) / beta_j
		mpi_times(R[j - 1], Q[j], 1.0f / beta[j - 1], N, rank, size, MPI_COMM_WORLD);
		// alpha_j = q_j * A * q_j
		cl.ApplyA(Q[j], temp_v1, MPI_COMM_WORLD);
		dot_sum = 0.0f;
		mpi_dot(temp_v1, Q[j], &dot_sum, N, rank, size, MPI_COMM_WORLD);
		alpha[j - 1] = dot_sum;
		// r_j = A * q_j - alpha_j * q_j - beta_j * q_(j-1)
		mpi_times(Q[j], temp_v2, alpha[j - 1], N, rank, size, MPI_COMM_WORLD);
		mpi_times(Q[j - 1], temp_v3, beta[j - 1], N, rank, size, MPI_COMM_WORLD);
		mpi_plus(temp_v1, temp_v2, temp_v4, N, rank, size, MPI_COMM_WORLD, false);
		mpi_plus(temp_v4, temp_v3, R[j], N, rank, size, MPI_COMM_WORLD, false);
		// beta_(j+1) = norm(r_j)
		dot_sum = 0.0f;
		mpi_dot(R[j], R[j], &dot_sum, N, rank, size, MPI_COMM_WORLD);
		beta[j] = sqrt(dot_sum);
	}


	// Output
	cout << "Alpha: [";
	for (int i = 0; i < K; i++)
		cout << alpha[i] << " ";
	cout << "]\n";

	cout << "Beta: [";
	for (int i = 0; i < K; i++)
		cout << beta[i] << " ";
	cout << "]\n";

	// Compute Eigenvalues with LAPACK
	lapack_int info = LAPACKE_sstev(LAPACK_ROW_MAJOR, 'N', K + 1, alpha, &beta[1], NULL, (lapack_int) 1);

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}