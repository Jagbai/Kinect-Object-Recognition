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
#include <io.h>
#include <string>
#include <direct.h>
#include "opencv2/features2d/features2d.hpp"


using namespace cv::ml;
using namespace cv;
using namespace std;
using namespace dnn;


FreenectPlaybackWrapper wrap("B:/Set1");


Mat newdepth;
Mat initDepth;

Mat Init_gray;
Mat rgbarr, backgrnd, detected_edges;
Mat initialframe;
Mat currentRGB;
Mat currentDepth;
Mat origdiff;
Mat newdiff;
Mat imggray;
std::string path;
Mat initthreshold;
Mat currentThreshold;

int lowThreshold = 100;
int classcounter = 0;
const int max_lowThreshold = 255;
const int finalratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
bool newfolder = false;

Mat translateImg(Mat &img, int x, int y) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}


void imgpreprocess(Mat & Thresimg, Mat & imgtosave)
{
	cv::Mat imgcrop;
	//normalize
	normalize(Thresimg, Thresimg, 0, 255, NORM_MINMAX, -1, Mat());
	//Histogram Equalization
	equalizeHist(Thresimg, Thresimg);
	//Translate the image into a 1 dimensional image array
	translateImg(Thresimg, -35, 30);

	// crop image and resize
	Rect roi(150, 70, 350, 250);
	resize(Thresimg(roi), imgtosave, Size(640, 480));
}

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
	// Calculate the differnce comparison
	cv::compare(mat1, mat2, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff);
	return nz == 0;
}

int main(int argc, char * argv[])


	//Create the RGB,Depth and intial Windows

	cv::namedWindow("RGB", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_GUI_EXPANDED);
	cv::namedWindow("Depth", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_GUI_EXPANDED);
	cv::namedWindow("InitialFrame", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_GUI_EXPANDED);
	//cv::namedWindow("Threshold", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_GUI_EXPANDED);

	//The key value represents a key pressed on the keyboard,
	//Where 27 is the ESC key
	//Create string array of labels for each category

	const std::string labels[] = { "Baby","Dog","Dinosaur","Coffee_Tin","Mug",
				 "Car","Camera","Keyboard","Koala","Blank","Blackberry",
				 "Diet_Coke_Bottle","Duck","Dragon","Android" };

	char key = '0';
	int framecounter = 0;

	//Initial background frame


	//Status represents curr status of the Playback wrapper
	//Value of 0 = finihsed playback

	// The status can by bitwise AND to determine if the RGB or
	// Depth image has been updated using the State enum.



	uint8_t status = 255;






	while (key != 27 && status != 0)
	{
		// Loads in the next frame of Kinect data into the
		// wrapper. Also pauses between the frames depending
		// on the time between frames.
		status = wrap.GetNextFrame();

		// Determine if RGB is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_RGB)
			currentRGB = wrap.RGB;

		// Determine if Depth is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_DEPTH)
			currentDepth = wrap.Depth;
		threshold(currentDepth, currentThreshold, 80, 255, THRESH_BINARY);
		currentThreshold.copyTo(newdepth);

		//If the frame counter is at the very start grabs the inital frame
		if (framecounter == 0) {
			initialframe = wrap.Depth;
			threshold(initialframe, initthreshold, 80, 255, THRESH_BINARY);
			initthreshold.copyTo(initDepth);
		}
		framecounter++;
		//Incremements frame counter


		//Create labels for image files
		String currRGB = ("UpdatedRGB" + std::to_string(framecounter) + ".jpg");
		String currDepth = ("Image" + std::to_string(framecounter) + ".jpg");

		
		/// Show the images of each type
		cv::imshow("InitialFrame", initDepth);
		cv::imshow("RGB", currentRGB);
		cv::imshow("Depth", newdepth);

		// If there is a difference in dpeth then create a new folder named with the label type
		if (!checkDiff(newdepth, initDepth) && newfolder == false) {
			cout << "TRUE" << endl;
			newfolder = true;
			if (newfolder) {
				path = "B:/TrainingData/" + labels[classcounter] + "/";
				cout << path << endl;
				if (classcounter != 9) {
					mkdir(path.c_str());
				}

				if (classcounter <= 13) {

					classcounter++;

				}

			}
		}
		// Stop storing files within the folder when there is no difference in depth anymore
		else if (checkDiff(newdepth, initDepth)) {
			cout << "FALSE" << endl;
			newfolder = false;

		}
		// Start storing images in label directory if there is a differnce in dpeth
		if (!checkDiff(newdepth, initDepth)) {
			
			cv::Mat imgtosave;
			imgpreprocess(newdepth, imgtosave);
			imwrite(path + currDepth, imgtosave);
			

		}

		// Check for keyboard input
		key = cv::waitKey(10);
	}

	return 0;
}



