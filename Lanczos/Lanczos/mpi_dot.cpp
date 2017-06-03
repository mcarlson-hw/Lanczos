#include <mpi.h>
#include<iostream>
using namespace std;

float* local_v1;
float* local_v2;
float* local_output;
float* sums;
float local_sum;

void mpi_dot(float* v1, float* v2, float* final_sum, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new float[N / size];
	local_v2 = new float[N / size];
	if (rank == 0)
	{
		sums = new float[size];
	}
	local_sum = 0.0f;

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_FLOAT, local_v1, N / size, MPI_FLOAT, 0, comm);
	MPI_Scatter(v2, N / size, MPI_FLOAT, local_v2, N / size, MPI_FLOAT, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_sum += local_v1[i] * local_v2[i];

	// Gather local sums
	MPI_Gather(&local_sum, 1, MPI_FLOAT, sums, 1, MPI_FLOAT, 0, comm);

	// Compute final sum
	if (rank == 0)
		for (int i = 0; i < size; i++)
			*final_sum += sums[i];

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] sums;
}
void mpi_plus(float* v1, float* v2, float* output, int N, int rank, int size, MPI_Comm comm, bool plusorminus)
{
	// Initialize local data
	local_v1 = new float[N / size];
	local_v2 = new float[N / size];
	local_output = new float[N / size];

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_FLOAT, local_v1, N / size, MPI_FLOAT, 0, comm);
	MPI_Scatter(v2, N / size, MPI_FLOAT, local_v2, N / size, MPI_FLOAT, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
	{
		if (plusorminus)
			local_output[i] = local_v1[i] + local_v2[i];
		else
			local_output[i] = local_v1[i] - local_v2[i];
	}

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_FLOAT, output, N / size, MPI_FLOAT, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] local_output;
}
void mpi_times(float* v, float* output, float scalar, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new float[N / size];
	local_output = new float[N / size];

	// Scatter v1 and v2 across processors
	MPI_Scatter(v, N / size, MPI_FLOAT, local_v1, N / size, MPI_FLOAT, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_output[i] = local_v1[i] * scalar;

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_FLOAT, output, N / size, MPI_FLOAT, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_output;
}