
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stddef.h>
#include<ctime>

using namespace cv;
using namespace std;
#include <string>

#define WINDOW_SIZE (3)
#define FILTER_SIZE (WINDOW_SIZE*WINDOW_SIZE)
#define TILE_SIZE 16
#define MEDIAN_INDEX (FILTER_SIZE/2)


__global__ void medianFilterKernel(unsigned char *inputImageKernel, unsigned char *outputImagekernel, int imageWidth, int imageHeight)
{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char filterVector[FILTER_SIZE];   //Take fiter window
	if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
		outputImagekernel[row*imageWidth + col] = 0; //Deal with boundry conditions
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++) {
				filterVector[x*WINDOW_SIZE + y] = inputImageKernel[(row + x - 1)*imageWidth + (col + y - 1)];   // setup the filterign window.
			}
		}
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = i + 1; j < FILTER_SIZE; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap the variables.
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImagekernel[row*imageWidth + col] = filterVector[MEDIAN_INDEX];   //Set the output variables.
	}
}

unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}


int main(int argc, char** argv)
{
	Mat src = imread("lenaSzum.jpeg", 1);
	Mat dst;

	int width = src.size().width;
	int height = src.size().height;

	unsigned char *sourceDataDevice, *filteredDataDevice;
	Mat source(src.size(), CV_8U, createImageBuffer(width * height, &sourceDataDevice));
	Mat filtered(src.size(), CV_8U, createImageBuffer(width * height, &filteredDataDevice));

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)width / (float)TILE_SIZE),
		(int)ceil((float)height / (float)TILE_SIZE));

		cvtColor(src, source, CV_BGR2GRAY);
		medianBlur(source, dst, 3);

		clock_t start = clock();

		medianFilterKernel << <dimGrid, dimBlock >> > (sourceDataDevice, filteredDataDevice, width, height);
		//cudaThreadSynchronize();
		cudaDeviceSynchronize();

		clock_t end = clock();
		clock_t time = double(end - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "Czas filtracji na GPU: " << time << "ms\n";

		imshow("source", source);
		imshow("filtered", filtered);
		imshow("filtered with opencv", dst);
		waitKey();
}
