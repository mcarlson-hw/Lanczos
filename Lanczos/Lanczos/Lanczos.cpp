#include <iostream>
#include "Lanczos.h"
using namespace std;

MPI_Comm global_comm;
int N;
float hx, hy, hz;
float a, ax, ay, az;
float *u_current, *u_old;

int ijk_to_m(int i, int j, int k)
{
	return 0;
}
int ijk_to_l(int i, int j, int k)
{
	return 0;
}
int m_to_r(int m)
{
	return 0;
}
int* m_to_ijk(int m)
{
	return 0;
}
int m_to_l(int m)
{
	return 0;
}

// Main Functions
void init(int dx, int dy, int dz)
{
	N = (1+dx)*(1+dy)*(1+dz);
	hx = 1.0f / (1.0f*dx);
	hy = 1.0f / (1.0f*dy);
	hz = 1.0f / (1.0f*dz);
	ax = 1.0f / (hx*hx);
	ay = 1.0f / (hy*hy);
	az = 1.0f / (hz*hz);
	a = 2.0f * (ax + ay + az);
	global_comm = MPI_COMM_WORLD;
	u_old = new float[N]();
	u_current = new float[N]();
}
void A(float* in, float* out, MPI_Comm comm)
{
	// Receive pieces of 'in' that are necessary to compute 'out'
}
void iterate(int n)
{

}