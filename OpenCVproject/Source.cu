
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
	VideoCapture camera(0);
	Mat frame;
	if (!camera.isOpened())
		return -1;


	return 0;
}
