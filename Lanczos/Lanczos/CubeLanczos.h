#pragma once
#include <mpi.h>
class CubeLanczos
{
private:
	// Constants
	const float PI = 3.1415927f;
	const float C = -0.008443431966f;
	const int periods[3] = { 0, 0, 0 };

	// Data
	float* OldData;
	float* NewData;
	float* top_neighbor;
	float* bottom_neighbor;
	float* left_neighbor;
	float* right_neighbor;
	float* front_neighbor;
	float* back_neighbor;
	float* top_data;
	float* bottom_data;
	float* left_data;
	float* right_data;
	float* front_data;
	float* back_data;

	// Parameters
	int n_rows, n_cols, n_layers, n_elems;

	float hx, hy, hz, h;
	float a, ax, ay, az, ah;
	int* divs;

	// MPI Stuff
	int p_id, n_processors, cart_rank;
	MPI_Comm cart_comm;
	int p_up, p_down, p_left, p_right, p_front, p_back;
	int* p_XYZ;
	MPI_Request up_r, down_r, left_r, right_r, front_r, back_r;
	MPI_Request up_s, down_s, left_s, right_s, front_s, back_s;

public:
	int* IJK;


	// Constructors
	CubeLanczos(int, int, int);
	CubeLanczos(int, int, int, int, int, int);

	// Internal Functions
	void ApplyA(float*, float*);
	void SwapBuffers();
	void PrepareOutgoingBuffers();

	// Static Functions
	void set_divs(int);
	void printData(int);

	// Coordinate Functions
	int ijk_to_m(int, int, int);
	int jk_to_m(int, int);
	int ij_to_m(int, int);
	int ik_to_m(int, int);
	void m_to_ijk(int);

	// Accessors
	float* data_pointer(int, int);

	// MPI
	void parallel_init();
	void communicate();
	void wait_for_sends();
	void wait_for_recvs();

	// Destructor
	//~CubeMesh();	// Free memory upon destruction
};