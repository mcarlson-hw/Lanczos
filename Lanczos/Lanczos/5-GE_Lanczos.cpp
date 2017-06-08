#include <iostream>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include "mpi_dot.h"
#include "CubeFD.h"
//#include "mkl.h"
using namespace std;
int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// ============================
	// == Main Execution Section ==
	// ============================

	// Seed RNG
	if (rank == 0) srand(time(NULL));

	// Declare Variables
	const int d = 30;
	const int N = d*d*d;
	const int K = 5;
	float** R;
	float** Q;
	float** P;
	float* alpha;
	float* beta;
	float dot_sum;
	float* temp_v1;

	// Allocate Memory
	R = new float*[K + 1];
	Q = new float*[K + 1];
	P = new float*[K + 2];
	for (int i = 0; i < K + 1; i++)
	{
		R[i] = new float[N]();
		Q[i] = new float[N]();
		P[i] = new float[N]();
	}
	P[K + 1] = new float[N]();
	alpha = new float[K + 1]();
	beta = new float[K + 2]();

	// Initialize
	for (int i = 0; i < N; i++)
		R[0][i] = ((float)rand()) / ((float)RAND_MAX);
	temp_v1 = new float[N];
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);

	// ====== Lanczos Non-Generalized Eigenvalue Solve ======
	// ======================================================

	// r0 is an already set random vector
	// p1 = M * r0
	cl.ApplyM(R[0], P[0], MPI_COMM_WORLD);
	// Set beta_1
	dot_sum = 0.0f;
	mpi_dot(R[0], P[0], &dot_sum, N, rank, size, MPI_COMM_WORLD);
	beta[0] = sqrt(dot_sum);

	// Main Loop
	for (int j = 1; j < K; j++)
	{
		// q_j = r_(j-1) / beta_(j-1)
		mpi_times(R[j - 1], Q[j], 1.0f / beta[j - 1], N, rank, size, MPI_COMM_WORLD);

		// p_(j-1) = p_(j-1) / beta_(j-1)
		mpi_times(P[j - 1], P[j - 1], 1.0f / beta[j - 1], N, rank, size, MPI_COMM_WORLD);

		// r_j = (K - sigma*M) \ p_(j-1)		(use MF_GMRES)
		// MF_GMRES

		// r_j = r_j - q_(j-1) * beta_(j-1)
		mpi_times(Q[j - 1], temp_v1, beta[j - 1], N, rank, size, MPI_COMM_WORLD);
		mpi_plus(R[j], temp_v1, R[j], N, rank, size, MPI_COMM_WORLD, false);

		// alpha_(j-1) = r_j (dot) p_(j-1)
		dot_sum = 0.0f;
		mpi_dot(R[j], P[j - 1], &dot_sum, N, rank, size, MPI_COMM_WORLD);
		alpha[j - 1] = dot_sum;

		// r_j = r_j - q_j * alpha_(j-1)
		mpi_times(Q[j], temp_v1, alpha[j - 1], N, rank, size, MPI_COMM_WORLD);
		mpi_plus(R[j], temp_v1, R[j], N, rank, size, MPI_COMM_WORLD, false);

		// p_j = M*r_j
		cl.ApplyM(R[j], P[j], MPI_COMM_WORLD);

		// beta_j = sqrt(r_j (dot) p_j)
		dot_sum = 0.0f;
		mpi_dot(R[j], P[j], &dot_sum, N, rank, size, MPI_COMM_WORLD);
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
	//lapack_int info = LAPACKE_sstev(LAPACK_ROW_MAJOR, 'N', K + 1, alpha, &beta[1], NULL, (lapack_int)1);

	// ======================================================

	// ===============================
	// == End Main Execution Region ==

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}