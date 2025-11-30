#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>

using namespace std;


inline int _ConvertSMVer2Cores(int major, int minor) {
	typedef struct {
		int SM;  // M = SM Major version,
				 // m = SM minor versionS
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x10,  8 }, //Tesla (SM 1.0) G80
		{ 0x11,  8 },
		{ 0x12,  8 },
		{ 0x13,  8 },
		{ 0x20, 32 },
		{ 0x21, 48 },
		{ 0x30, 192 },
		{ 0x35, 192 },
		{ -1, -1 } //none of the above/ default/ error
	};

	int i = 0;

	while (nGpuArchCoresPerSM[i].SM != -1) {
		if (nGpuArchCoresPerSM[i].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[i].Cores;
		}

		i++;
	}
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[i - 1].Cores);
	return nGpuArchCoresPerSM[i - 1].Cores;
}


int main() {

	int device_count;
	cudaGetDeviceCount(&device_count);
	cudaDeviceProp device_Properties;
	cout << "Number of Devices: " << device_count << endl;

	for (int i = 0; i < device_count; i++) {
		cudaGetDeviceProperties(&device_Properties, i);
		
		cout << "Type of CUDA Device: " << device_Properties.name << endl;
		cout << "Device Number:  " << i + 1 << endl;
		cout << "Clock Rate: " << device_Properties.clockRate << endl;
		cout << "Number of Streaming Multiprocessors: " << device_Properties.multiProcessorCount << endl;
		cout << "Number of Cores: " << _ConvertSMVer2Cores(device_Properties.major, device_Properties.minor) << endl;
		cout << "Warp Size: " << device_Properties.warpSize << endl;
		cout << "Amount of Global Memory: " << device_Properties.totalGlobalMem << endl;
		cout << "Amount of Constant Memory: " << device_Properties.totalConstMem << endl;
		cout << "Amount of Shared Memory Per Block: " << device_Properties.sharedMemPerBlock << endl;
		cout << "Registers Available Per Block: " << device_Properties.regsPerBlock << endl;
		cout << "Max Threads Per Block: " << device_Properties.maxThreadsPerBlock << endl;
		cout << "Max Dimension Size of Block: " << endl;
		cout << "X: " << device_Properties.maxThreadsDim[0] << "      Y: " << device_Properties.maxThreadsDim[1] << "		Z: " << device_Properties.maxThreadsDim[2] << endl;
		cout << "Max Dimension Size of Grid: " << endl;
		cout << "X: " << device_Properties.maxGridSize[0] << "		Y: "<< device_Properties.maxGridSize[1] << "	Z: " << device_Properties.maxGridSize[2] << endl;
		



	}

}