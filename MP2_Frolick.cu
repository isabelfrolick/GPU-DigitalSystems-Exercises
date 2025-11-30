#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define SIZE 256

__global__ void kernel_addition(int *A, int *B, int *C, int num)
{
	//Each thread produces one output matrix element
	/*
	int i = (blockIdx.x *blockDim.x + threadIdx.x) * num + (blockIdx.y *blockDim.y + threadIdx.y);
	C[i] = A[i]+B[i];

	*/


	/*	//Each thread produces one output matrix row
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num) {
	for (int k = 0; k < num; k++) {
	C[i*num + k] = B[i*num + k] + A[i*num + k];
	}
	}*/



	//Each thread produces one output matrix column
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (i<num) {
		for (int k = 0; k<num; k++) {
			C[i*num + k] = A[i*num + k] + B[i*num + k];
		}
	}


}

cudaError_t matrix_addition(int N[SIZE][SIZE], int M[SIZE][SIZE], int P[SIZE][SIZE], int num);

int main() {
	auto matrix1 = new int[SIZE][SIZE];
	auto matrix2 = new int[SIZE][SIZE];
	auto matrix_out = new int[SIZE][SIZE];

	srand(time(NULL));

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			matrix1[i][j] = (int)rand() / 10;
			matrix2[i][j] = (int)rand() / 10;
		}
	}

	matrix_addition(matrix1, matrix2, matrix_out, SIZE);
	free(matrix1);
	free(matrix2);
	free(matrix_out);
}

cudaError_t matrix_addition(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE], int num) {

	auto matrix1 = new int[SIZE][SIZE];
	auto matrix2 = new int[SIZE][SIZE];
	auto matrix_out = new int[SIZE][SIZE];
	auto CPU_matrix_out = new int[SIZE][SIZE];
	bool fail = false;

	int square_matrix_size = num*num;

	for (int x = 0; x < num; x++) {
		for (int y = 0; y < num; y++) {
			CPU_matrix_out[x][y] = 0;
		}
	}

	//each thread producing one output matrix element
	//dim3 threads(SIZE, SIZE, 1);
	//dim3 blocks(num / SIZE, num / SIZE, 1);

	//each thread producing one output matrix row
	//dim3 threads = dim3(num/SIZE, 1, 1);
	//dim3 blocks = dim3(16, 1, 1);

	//each thread producing one output matrix column
	dim3 threads(1, num / SIZE, 1);
	dim3 blocks(1, SIZE, 1);


	//checking if we can use the first device
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three matrices (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&matrix1, square_matrix_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&matrix2, square_matrix_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&matrix_out, square_matrix_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//create event for timing purposes 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	float timing_GPU = 0.0f;

	cudaEventRecord(start, 0);

	cudaStatus = cudaMemcpy(matrix1, A, square_matrix_size* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemcpy failed for Input Matrix 1!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matrix2, B, square_matrix_size* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemcpy failed for Input Matrix 2!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matrix_out, C, square_matrix_size* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemcpy failed for Output Matrix!");
		goto Error;
	}


	kernel_addition << <threads, blocks >> >(*matrix1, *matrix2, *matrix_out, num);


	cudaStatus = cudaMemcpy(C, matrix_out, square_matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMemcpy failed for Output Matrix returning from device to host!");
		goto Error;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timing_GPU, start, stop);

	cout << "The GPU took: " << timing_GPU / 1000 << " seconds." << endl;



	clock_t cpuStart = clock();
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < num; j++) {
			CPU_matrix_out[i][j] = A[i][j] + B[i][j];
			if (CPU_matrix_out[i][j] != C[i][j]) {
				cout << "Test failed.";
				fail = true;
				break;
			}

		}
	}
	if (!fail) cout << "Test PASSED!" << endl;

	float finish = (float)(clock() - cpuStart) / CLOCKS_PER_SEC;

	cout << "The CPU took: " << finish << " seconds." << endl;

	return cudaStatus;
Error:
	free(matrix1);
	free(matrix2);
	free(matrix_out);
	return cudaStatus;
}