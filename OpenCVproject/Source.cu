
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

#ifndef WINDOW_SIZE
#define WINDOW_SIZE (5)
#endif

#define TILE_SIZE 16

__global__ void convolveKernel(unsigned const char* src, unsigned char* dist) {

}

__global__ void medianFilterKernel(unsigned char *inputImageKernel, unsigned char *outputImagekernel, int imageWidth, int imageHeight)
{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char filterVector[9] = { 0,0,0,0,0,0,0,0,0 };   //Take fiter window
	if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
		outputImagekernel[row*imageWidth + col] = 0; //Deal with boundry conditions
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++) {
				filterVector[x*WINDOW_SIZE + y] = inputImageKernel[(row + x - 1)*imageWidth + (col + y - 1)];   // setup the filterign window.
			}
		}
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap the variables.
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImagekernel[row*imageWidth + col] = filterVector[4];   //Set the output variables.
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
	VideoCapture camera(0);
	Mat frame;
	if (!camera.isOpened())
		return -1;

	camera >> frame;
	//IplImage* img = cvLoadImage("lena.jpg", 1);
	//IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	//double a[9] = { 1.0, 2.0, 1.0,
	//	0.0, 0.0, 0.0,
	//	-1.0, -2.0, -1.0};
	//CvMat k;
	//cvInitMatHeader(&k, 3, 3, CV_64FC1, a);

	//cvFilter2D(img, dst, &k, cvPoint(-1, -1));
	//cvSaveImage("filtered.jpg", dst);

	//namedWindow("Before", CV_WINDOW_AUTOSIZE);

	//Mat src = imread("lena.jpg", 1);

	Mat dst;


	//imshow("Before", src);

	//medianBlur(src, dst, 7);

	//imshow("Median filter", dst);


	int width = frame.size().width;
	int height = frame.size().height;

	unsigned char *sourceDataDevice, *filteredDataDevice;
	Mat source(frame.size(), CV_8U, createImageBuffer(width * height, &sourceDataDevice));
	Mat filtered(frame.size(), CV_8U, createImageBuffer(width * height, &filteredDataDevice));

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)width / (float)TILE_SIZE),
		(int)ceil((float)height / (float)TILE_SIZE));

	while (1)
	{
		camera >> frame;
		cvtColor(frame, source, CV_BGR2GRAY);
		clock_t start = clock();
		medianBlur(source, dst, 1);
		clock_t end = clock();
		double time = double(end - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "Czas filtracji na CPU: " << time << "ms\n";

		start = clock();

		medianFilterKernel << <dimGrid, dimBlock >> > (sourceDataDevice, filteredDataDevice, width, height);
		//cudaThreadSynchronize();
		cudaDeviceSynchronize();

		end = clock();
		time = double(end - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "Czas filtracji na GPU: " << time << "ms\n";

		imshow("source", source);
		imshow("filtered", filtered);
		imshow("filtered with opencv", dst);
		if (cv::waitKey(1) == 27) break;
	}
}
