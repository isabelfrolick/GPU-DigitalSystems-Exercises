#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define SIZE 32
#define BLOCK_WIDTH 16

__global__ void kernel_multiplication(int *N, int *M, int *P, int nums)
{

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < nums && COL < nums) {
		int sum = 0;
		for (int k = 0; k < nums; k++) 
			sum += N[ROW*nums + k] * M[k*nums + COL];

		P[ROW*nums + COL] = sum;
	}

}

cudaError_t multiply_matrices(int N[SIZE][SIZE], int M[SIZE][SIZE], int P[SIZE][SIZE], int nums);

int main() {
	auto matrix1 = new int[SIZE][SIZE];
	auto matrix2 = new int[SIZE][SIZE];
	auto output_matrix = new int[SIZE][SIZE];


	srand(time(NULL));

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			matrix1[i][j] = (int)rand()/10;
			matrix2[i][j] = (int)rand()/10;
			output_matrix[i][j] = 0;
			
		}
	}

	multiply_matrices(matrix1, matrix2, output_matrix, SIZE);

	free(matrix1);
	free(matrix2);
	free(output_matrix);
}

cudaError_t multiply_matrices(int N[SIZE][SIZE], int M[SIZE][SIZE], int P[SIZE][SIZE], int nums) {
	auto matrix1 = new int[SIZE][SIZE];
	auto matrix2 = new int[SIZE][SIZE];
	auto output_matrix = new int[SIZE][SIZE];
	auto CPU_output_matrix = new int[SIZE][SIZE];
	bool fail = false;

	int squared_num_elements = nums*nums;

	for (int x = 0; x < nums; x++) {
		for (int y = 0; y < nums; y++) {
			CPU_output_matrix[x][y] = 0;

		}
	}

	int blocksPerThread = SIZE / BLOCK_WIDTH;
	if (SIZE%BLOCK_WIDTH) blocksPerThread++;

	dim3 threads(blocksPerThread, blocksPerThread);
	dim3 blocks(BLOCK_WIDTH, BLOCK_WIDTH);



	//checking if we can use the first device
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three matrices (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&matrix1, squared_num_elements * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&matrix2, squared_num_elements * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&output_matrix, squared_num_elements * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	float elasped_GPU_time = 0.0f;

	cudaEventRecord(start, 0);

	cudaStatus = cudaMemcpy(matrix1, N, squared_num_elements* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for input one!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matrix2, M, squared_num_elements* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for input two!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(output_matrix, P, squared_num_elements* sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for output!");
		goto Error;
	}


	kernel_multiplication << <threads, blocks >> >(*matrix1, *matrix2, *output_matrix, nums);





	cudaStatus = cudaMemcpy(P, output_matrix, squared_num_elements * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for output coming back!");
		goto Error;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elasped_GPU_time, start, stop);

	cout << "The GPU took: " << elasped_GPU_time / 1000 << " seconds" << endl;



	clock_t start_CPU_time = clock();
	for (int i = 0; i < nums; i++) {
		for (int j = 0; j < nums; j++) {
			for (int k = 0; k < nums; k++) {
				CPU_output_matrix[i][j] += N[i][k] * M[k][j];
			}
			if (CPU_output_matrix[i][j] != P[i][j]) {
				cout << "Test failed.";
				cout << CPU_output_matrix[i][j] << ", ";
				cout << i << ", " << j <<endl;
				fail = true;
				break;
			}

		}
	}
	if (!fail) cout << "Test PASSED" << endl;

	float finish = (float)(clock() - start_CPU_time) / CLOCKS_PER_SEC;

	cout << "The CPU took " << finish << " seconds";

	return cudaStatus;
Error:
	free(matrix1);
	free(matrix2);
	free(output_matrix);
	return cudaStatus;
}