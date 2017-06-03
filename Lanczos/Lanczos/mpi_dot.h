#pragma once
#include <mpi.h>
void mpi_dot(float*, float*, float*, int, int, int, MPI_Comm);
void mpi_plus(float*, float*, float*, int, int, int, MPI_Comm, bool);
void mpi_times(float*, float*, float, int, int, int, MPI_Comm);