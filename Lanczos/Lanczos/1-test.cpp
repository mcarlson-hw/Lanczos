#include <iostream>
#include <mpi.h>
#include "CubeLanczos.h"
using namespace std;
int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int d = 2;
	CubeLanczos cl(d, d, d, rank, size);

	float* input;
	input = new float[d*d*d];
	for (int i = 0; i < d*d*d; i++)
		input[i] = i + 1;
	
	float* output;
	output = new float[d*d*d];

	// Test applying A to u
	cl.ApplyA(input, output);

	cout << "Input Vector: [";
	for (int i = 0; i < d*d*d; i++)
		cout << input[i] << " ";
	cout << "]\n";

	cout << "Output Vector: [";
	for (int i = 0; i < d*d*d; i++)
		cout << output[i] << " ";
	cout << "]\n";

	return 0;
}

