#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <map>
#include <string>
#include <stack>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>

#include "Common/helper_cuda.h"
#include "Common/helper_string.h"
#include "Common/helper_functions.h"
#include "Common/helper_image.h"

using namespace std;

typedef float2 Complex;
static __device__ __host__ inline Complex Add(Complex, Complex);
static __device__ __host__ inline Complex Scale(Complex, float);
static __device__ __host__ inline Complex Mul(Complex, Complex);
static __global__ void PointwiseMulAndScale(Complex*, const Complex*, int, float);


static __device__ __host__ inline Complex Add(Complex a, Complex b) {
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

static __device__ __host__ inline Complex Scale(Complex a, float s) {
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __device__ __host__ inline Complex Mul(Complex a, Complex b) {
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

static __global__ void PointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		a[i] = Scale(Mul(a[i], b[i]), scale);
	}
}

void Convolve(const Complex*, int, const Complex*, int, Complex*);

int PadData(const Complex*, Complex**, int, const Complex*, Complex**, int);

void runTest();

void save(string, const Complex*, int);

std::string info();

#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11


int main(int argc, char** argv) {
	setlocale(LC_ALL, "Russian");
	cout << info() << endl;
	runTest();
}

void save(string filename, Complex* data, int size) {
	FILE* fp;

	const int SIZE = filename.length() + 1;

	char* char_array = new char[SIZE];

	// copying the contents of the
	// string to char array
	strcpy(char_array, filename.c_str());

	fp = fopen(char_array, "w");

	for (int i = 0; i < size; i++) {
		fflush(stdin);
		fprintf(fp, "X: %f  Y: %f\n", data[i].x, data[i].y);
	}

	fclose(fp);
}

std::string info() {
	std::string str;
	int devices;
	cudaDeviceProp info;
	cudaGetDeviceCount(&devices);

	str = "Количество GPU поддерживаемых CUDA: ";
	str += std::to_string(devices);
	str += ";\n";

	for (int i = 0; i < devices; i++)
	{
		cudaGetDeviceProperties(&info, i);
		str += "Название GPU: ";
		str += info.name;
		str += ";\n";
		str += "Доступная память: ";
		str += std::to_string(info.totalGlobalMem / 1048576);
		str += " MB";
		str += ";\n";
		str += "Доступная постоянная память память: ";
		str += std::to_string(info.totalConstMem);
		str += " B";
		str += ";\n";
		str += "Общая память для блоков: ";
		str += std::to_string(info.sharedMemPerBlock);
		str += " B";
		str += ";\n";
		str += "Общее количество 32 - битных регистров: ";
		str += std::to_string(info.regsPerBlock);
		str += ";\n";
		str += "Размер Warp: ";
		str += std::to_string(info.warpSize);
		str += ";\n";
		str += "Максимальное количество потоков в блоке: ";
		str += std::to_string(info.maxThreadsPerBlock);
		str += ";\n";
		str += "Максимальный размер блока: ";
		str += std::to_string(info.maxThreadsDim[0]);
		for (int i = 1; i < 3; i++) {
			str += "x";
			str += std::to_string(info.maxThreadsDim[i]);

		}
		str += ";\n";
		str += "Максимальный размер сетки: ";
		str += std::to_string(info.maxGridSize[0]);
		for (int i = 1; i < 3; i++) {
			str += "x";
			str += std::to_string(info.maxGridSize[i]);

		}
		str += ";\n";
		str += "Тактовая частота: ";
		str += std::to_string(info.clockRate / 1000);
		str += " MHz";
		str += ";\n";
		str += "Частота шины: ";
		str += std::to_string(info.memoryClockRate / 1000);
		str += " MHz";
		str += ";\n";
		str += "Ширина шины: ";
		str += std::to_string(info.memoryBusWidth);
		str += ";\n";
		str += "Кэш l2: ";
		str += std::to_string(info.l2CacheSize);
		str += " B";
		str += ";\n";
	}
	return str;
}


void runTest() {

	Complex* h_signal =
		reinterpret_cast<Complex*>(malloc(sizeof(Complex) * SIGNAL_SIZE));

	// Инициализировать память для сигнала
	for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
		h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
		h_signal[i].y = 0;
	}


	Complex* h_filter_kernel =
		reinterpret_cast<Complex*>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));

	// Инициализируйте память для фильтра
	for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
		h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX);
		h_filter_kernel[i].y = 0;
	}

	// Ядро сигнала pad и фильтра
	Complex* h_padded_signal;
	Complex* h_padded_filter_kernel;
	int new_size =
		PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
			&h_padded_filter_kernel, FILTER_KERNEL_SIZE);
	int mem_size = sizeof(Complex) * new_size;


	Complex* d_signal;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));
	checkCudaErrors(
		cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));


	Complex* d_filter_kernel;
	checkCudaErrors(
		cudaMalloc(reinterpret_cast<void**>(&d_filter_kernel), mem_size));


	checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
		cudaMemcpyHostToDevice));

	cufftHandle plan;
	checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

	cufftHandle plan_adv;
	size_t workSize;
	long long int new_size_long = new_size;

	checkCudaErrors(cufftCreate(&plan_adv));
	checkCudaErrors(cufftXtMakePlanMany(plan_adv, 1, &new_size_long, NULL, 1, 1,
		CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1,
		&workSize, CUDA_C_32F));
	cout << "Размер буфера " << workSize << " bytes" << endl;

	// Преобразование сигнала и ядра
	cout << "Transforming signal ..." << endl;
	checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
		reinterpret_cast<cufftComplex*>(d_signal),
		CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(
		plan_adv, reinterpret_cast<cufftComplex*>(d_filter_kernel),
		reinterpret_cast<cufftComplex*>(d_filter_kernel), CUFFT_FORWARD));

	// Умножение коэффициентов и нормализация результата
	cout << "Launching PointwiseMulAndScale ..." << endl;
	PointwiseMulAndScale << <32, 256 >> > (d_signal, d_filter_kernel, new_size, 1.0f / new_size);

	getLastCudaError("Kernel execution failed [ PointwiseMulAndScale ]");

	// Преобразование сигнала обратно
	cout << "Transforming signal back ..." << endl;
	checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal), reinterpret_cast<cufftComplex*>(d_signal), CUFFT_INVERSE));

	Complex* h_convolved_signal = h_padded_signal;
	checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

	// Память хоста для результата свертки
	Complex* h_convolved_signal_ref = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * SIGNAL_SIZE));

	// Свертка
	Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE, h_convolved_signal_ref);

	bool bTestResult = sdkCompareL2fe(
		reinterpret_cast<float*>(h_convolved_signal_ref),
		reinterpret_cast<float*>(h_convolved_signal), 2 * SIGNAL_SIZE, 1e-5f);

	save("h_signal.txt", h_signal, SIGNAL_SIZE);
	save("h_filter_kernel.txt", h_filter_kernel, FILTER_KERNEL_SIZE);
	save("h_padded_signal.txt", h_padded_signal, 56);
	save("h_convolved_signal_ref.txt", h_convolved_signal_ref, SIGNAL_SIZE);

	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(plan_adv));
	free(h_signal);
	free(h_filter_kernel);
	free(h_padded_signal);
	free(h_padded_filter_kernel);
	free(h_convolved_signal_ref);
	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(d_filter_kernel));

	exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int PadData(const Complex* signal, Complex** padded_signal, int signal_size, const Complex* filter_kernel, Complex** padded_filter_kernel, int filter_kernel_size) {
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	Complex* new_data =
		reinterpret_cast<Complex*>(malloc(sizeof(Complex) * new_size));
	memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
	*padded_signal = new_data;

	new_data = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * new_size));
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
	memset(new_data + maxRadius, 0,
		(new_size - filter_kernel_size) * sizeof(Complex));
	memcpy(new_data + new_size - minRadius, filter_kernel,
		minRadius * sizeof(Complex));
	*padded_filter_kernel = new_data;

	return new_size;
}

void Convolve(const Complex* signal, int signal_size, const Complex* filter_kernel, int filter_kernel_size, Complex* filtered_signal) {
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;

	for (int i = 0; i < signal_size; ++i) {
		filtered_signal[i].x = filtered_signal[i].y = 0;

		for (int j = -maxRadius + 1; j <= minRadius; ++j) {
			int k = i + j;

			if (k >= 0 && k < signal_size) {
				filtered_signal[i] = Add(filtered_signal[i], Mul(signal[k], filter_kernel[minRadius - j]));
			}
		}
	}
}
