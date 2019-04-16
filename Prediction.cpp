
#include "freenect-playback-wrapper.h"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <direct.h>
#include <io.h>
#include <string>
#include <vector>
#define CV_AA   16

using namespace cv::ml;
using namespace cv;
using namespace std;

/////**************Input parameter**************///////
string Videopath = "B:/Set2";

string Modelpath = "B:/Model.xml";
string savefolder = "B:/predictimage/";

//**********************************************///


HOGDescriptor hog(
	Size(640, 480), //winSize
	Size(8, 8), //blocksize
	Size(8, 8), //blockStride,
	Size(8, 8), //cellSize,
	9, //nbins,
	1, //derivAper,
	-1, //winSigma,
	HOGDescriptor::L2Hys, //histogramNormType,
	0.2, //L2HysThresh,
	0,//gammal correction,
	64,//nlevels=64
	1);

Ptr<SVM> svm = SVM::create();
std::vector<std::vector<float> > imgHOG;
vector<float> descriptors;
Mat newdepth;
Mat initDepth;
Mat initialframe;
Mat initthreshold;
Mat currentThreshold;
bool newfolder = false;
int classcounter = 0;
std::string path = "B:/ResultsData/";
//Create string array of labels for each category

string labels[] = { "Baby","Dog","Dinosaur","Coffee_Tin","Mug",
"Car","Camera","Keyboard","Koala","Blackberry",
"Diet_Coke_Bottle","Duck","Dragon","Android" };


// declare functions

void imgtext(Mat & frame, int response);
Mat translateImg(Mat &img, int offsetx, int offsety);
void imgpreprocess(Mat & Thresimg, Mat & imgcrop);

bool checkDiff(const cv::Mat mat1, const cv::Mat mat2)
{
	// treat two empty mat as identical as well
	if (mat1.empty() && mat2.empty()) {
		return true;
	}
	// if dimensionality of two mat is not identical, these two mat is not identical
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
		return false;
	}
	cv::Mat diff;
	cv::compare(mat1, mat2, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff);
	return nz == 0;
}


//Put the response text onto the screen next to the object
void imgtext(Mat & frame, int response)
{
	switch (response) {
	case 0:
	{
		putText(frame, "Baby", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 1:
	{
		putText(frame, "Dog", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 2:
	{
		putText(frame, "Dinosaur", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 3:
	{
		putText(frame, "Coffee_Tin", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 4:
	{
		putText(frame, "Mug", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 5:
	{
		putText(frame, "Car", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 6:
	{
		putText(frame, "Camera", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 7:
	{
		putText(frame, "Keyboard", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 8:
	{
		putText(frame, "Koala", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 9:
	{
		putText(frame, "Blackberry", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 10:
	{
		putText(frame, "Coke_Bottle", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 11:
	{
		putText(frame, "Duck", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 12:
	{
		putText(frame, "Dragon", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 13:
	{
		putText(frame, "Android", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	default: printf(" ");
	}
}


void imgpreprocess(Mat & Thresimg, Mat & imgtosave)
{
	cv::Mat imgcrop;
	//normalize
	normalize(Thresimg, Thresimg, 0, 255, NORM_MINMAX, -1, Mat());
	//Histogram Equalization
	equalizeHist(Thresimg, Thresimg);
	//Translate image into a 1 dimensional matrix
	translateImg(Thresimg, -35, 30);

	// crop image and resize
	Rect roi(150, 70, 350, 250);
	resize(Thresimg(roi), imgtosave, Size(640, 480));
}
Mat translateImg(Mat &img, int offsetx, int offsety) {
	//Translate image
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	//Apply affine transformation to the image
	warpAffine(img, img, trans_mat, img.size());
	return img;
}

int main(int argc, char * argv[])
{

	FreenectPlaybackWrapper wrap(Videopath);

	cv::Mat currentRGB;
	cv::Mat currentDepth;
	cv::Mat currentThreshold;
	cv::Mat currentContour;
	cv::Mat Results(cv::Size(640, 640), CV_8UC3);
	// Create the RGB and Depth Windows
	cv::namedWindow("Results", WINDOW_NORMAL);

	char key = '0';
	int framecounter = 0;
	uint8_t status = 255;

	// svm loading
	svm = SVM::load(Modelpath);
	while (key != 27 && status != 0)
	{
		status = wrap.GetNextFrame();
		//Grab the RGB from the wrap
		if (status & State::UPDATED_RGB)
			currentRGB = wrap.RGB;
		if (status & State::UPDATED_DEPTH)
		{
			//Grab depth from the wrap and preprocess
			currentDepth = wrap.Depth;
			threshold(currentDepth, currentThreshold, 80, 255, THRESH_BINARY);
			currentThreshold.copyTo(newdepth);
			
			
		}
		if (framecounter == 0) {
			//Grab the initial frame by checking if the framecounter is at the beginning
			initialframe = wrap.Depth;
			threshold(initialframe, initthreshold, 80, 255, THRESH_BINARY);
			initthreshold.copyTo(initDepth);
		}
		//Increment frame counter
		framecounter++;

		//Create labels		
		String currDepth = ("Image" + std::to_string(framecounter) + ".jpg");

		//Create new directories for each label in the string array of labels
		for (int i = 0; i < 13; i++) {

			string label = labels[i];

			string newdir = path + label + "/";
			mkdir(newdir.c_str());
		}			   		 

		// If there is a difference in dpeth then
		if (!checkDiff(newdepth, initDepth)) {

			cv::Mat imgtosave;
			//Preprocess the image
			imgpreprocess(newdepth, imgtosave);
			
			//Compute the descriptors 
			hog.compute(imgtosave, descriptors);
			//Create a response prediction
			int response = svm->predict(descriptors);
			//Put the response text into the image
			imgtext(currentRGB, response);
			//Write the file to the correct directory
			imwrite(path + "/" + labels[response] + "/" + currDepth, imgtosave);			
			
		}		   		 	  	  	   

		// Show the images in one windows
		cv::Mat RGB_mat;
		cv::Mat Depth_mat;
		cv::Mat Threshold_mat;
		cv::Mat Contour_mat;
		cv::Mat Gray_mat;

		currentRGB.copyTo(Results(cv::Rect(0, 0, 640, 480)));

		resize(currentDepth, Depth_mat, Size(currentDepth.cols / 3, currentDepth.rows / 3));
		cvtColor(Depth_mat, Depth_mat, COLOR_GRAY2BGR);
		Depth_mat.copyTo(Results(cv::Rect(0, 480, Depth_mat.cols, Depth_mat.rows)));

		resize(currentThreshold, Threshold_mat, Size(currentThreshold.cols / 3, currentThreshold.rows / 3));
		cvtColor(Threshold_mat, Threshold_mat, COLOR_GRAY2BGR);
		Threshold_mat.copyTo(Results(cv::Rect(currentDepth.cols / 3, 480, currentThreshold.cols / 3, currentThreshold.rows / 3)));

		

		cv::imshow("Results", Results);
		// Check for keyboard input
		key = cv::waitKey(10);
	}
	return 0;
}
