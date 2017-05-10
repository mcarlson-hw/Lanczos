#pragma once
#include <mpi.h>

// Coordinate Maps:
//		Grid Coordinate:			 (i,j,k)
//		Global Vector u Coordinate:  (m)
//		Rank:						 (r)
//		Local Vector u_r Coordinate: (l)
int ijk_to_m(int i, int j, int k);
int ijk_to_l(int i, int j, int k);
int m_to_r(int m);
int* m_to_ijk(int m);
int m_to_l(int m);

// Main Functions
void init(int dx, int dy, int dz);
void A(float* in, float* out, MPI_Comm comm);
void iterate(int n);