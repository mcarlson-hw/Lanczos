//#include <iostream>
//#include <mpi.h>
//#include "CubeFD.h"
//using namespace std;
//int main(int argc, char **argv)
//{
//	MPI_Init(&argc, &argv);
//
//	int rank, size;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//	int d = 3;
//	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);
//
//	float* input;
//	float* output;
//	if (rank == 0)
//	{
//		input = new float[d*d*d];
//		for (int i = 0; i < d*d*d; i++)
//			input[i] = i + 1;
//
//		output = new float[d*d*d];
//	}
//
//	// Test applying A to u
//	cl.ApplyA(input, output, MPI_COMM_WORLD);
//
//	if (rank == 0)
//	{
//		cout << "Input Vector: [";
//		for (int i = 0; i < d*d*d; i++)
//			cout << input[i] << " ";
//		cout << "]\n";
//
//		cout << "Output Vector: [";
//		for (int i = 0; i < d*d*d; i++)
//			cout << output[i] << " ";
//		cout << "]\n";
//	}
//
//
//	MPI_Finalize();
//
//	return 0;
//}