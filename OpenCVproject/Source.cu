
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

using namespace cv;
using namespace std;
#include <string>


int main(int argc, char** argv)
{
/*	VideoCapture camera(0);
	Mat frame;
	if (!camera.isOpened())
	*/	//return -1;

	IplImage* img = cvLoadImage("lena.jpg", 1);
	IplImage* dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	double a[9] = { 1.0, 2.0, 1.0,
		0.0, 0.0, 0.0,
		-1.0, -2.0, -1.0};
	CvMat k;
	cvInitMatHeader(&k, 3, 3, CV_64FC1, a);

	cvFilter2D(img, dst, &k, cvPoint(-1, -1));
	cvSaveImage("filtered.jpg", dst);
}
