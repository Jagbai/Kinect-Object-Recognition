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
#include <string>
#include <vector>
#include "opencv2/objdetect.hpp"

using namespace cv::ml;
using namespace cv;
using namespace std;

//**************parameter********************//
string modelpath = "B:/Model.xml";

//*******************************************//

struct DataSet { std::string filename; int label; };

vector<DataSet> dataList;
typedef std::vector<std::string> stringvec;

vector<DataSet> datalists();
void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG);
void ConvertVectortoMatrix(std::vector<std::vector<float> > &ipHOG, Mat & opMat);
void ConvertlabeltoMat(vector<int>  &iplabel, Mat &oplabel);
void SVMtrain(Mat &trainMat, Mat &trainLabels);

void read_directory(const std::string& name, stringvec& v);

int s = 0;
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

//Initialise array of strings
string labels[] = { "Baby","Dog","Dinosaur","Coffee_Tin","Mug",
"Car","Camera","Keyboard","Koala","Blackberry",
"Diet_Coke_Bottle","Duck","Dragon","Android" };

//Initialise data lists
vector<DataSet> datalists() {
	for (int j = 0; j < 13; j++)
	{
		// Read all the files within directory of labelled training data
		std::vector<std::string> filenames;
		string folderName;
		folderName = "B:/TrainingData/" +(labels[j]);
		read_directory(folderName, filenames);

		for (int jj = 0; jj < filenames.size(); jj++)
		{
			//Create data array with training data found in the directory
			DataSet tempDataset;
			tempDataset.filename = "B:/TrainingData/" +(labels[j])+"/"+ filenames[jj];
			tempDataset.label = j;
			dataList.push_back(tempDataset);
		}
	};
	return dataList;

};
//  get the name of all files in folder
void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*.jpg");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	//If the program finds a file that matches the pattern it pushes to vector 
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

//Compute the descriptor for the depth image using Histogram of Oriented Gradietns

void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG) {

	for (int y = 0; y < inputCells.size(); y++) {
		vector<float> descriptors;
		hog.compute(inputCells[y], descriptors);
		outputHOG.push_back(descriptors);
	}
}
//Convert the descriptor vector into a one dimenisonal matrix ready for training within the support vector machine
void ConvertVectortoMatrix(std::vector<std::vector<float> > &ipHOG, Mat &opMat)
{

	int descriptor_size = ipHOG[0].size();

	for (int i = 0; i < ipHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			opMat.at<float>(i, j) = ipHOG[i][j];
		}
	}
};
//Convert label to matrix ready for training within the support vector machine 
void ConvertlabeltoMat(vector<int>  &iplabel, Mat &oplabel)
{

	for (int i = 0; i < iplabel.size(); i++)
	{

		oplabel.at<int>(i, 0) = iplabel[i];
	}
};
// Use training data within support vector machine  to train a support vector machine and then save model within the model path

void SVMtrain(Mat &trainMat, Mat &labelsMat) {
	Ptr<SVM> svm = SVM::create();
	svm->setGamma(0.001);
	svm->setC(100);
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainMat, ROW_SAMPLE, labelsMat);
	svm->save(modelpath);
}

int main(int argc, char * argv[])
{

	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	vector<DataSet> datalist = datalists();
	//Create training data structure
	for (int l = 0; l < datalist.size(); l++)
	{
		Mat img = imread(datalist[l].filename, 0);
		trainCells.push_back(img);
		trainLabels.push_back(datalist[l].label);
	};

	std::vector<std::vector<float> > trainHOG;

	//Compute descriptor 
	computeHOG(trainCells, trainHOG);

	int descriptor_size = trainHOG[0].size();

	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
	//Convert vector to matrix
	ConvertVectortoMatrix(trainHOG, trainMat);

	Mat labelsMat(datalist.size(), 1, CV_32SC1);
	ConvertlabeltoMat(trainLabels, labelsMat);
	// Train the SVM
	SVMtrain(trainMat, labelsMat);

	cout << "Finished Training" << endl;
	system("pause");

	return 0;
}
