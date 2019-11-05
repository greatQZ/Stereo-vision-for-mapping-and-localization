#include<iostream>   
#include <opencv2/opencv.hpp>
#include "elas.h"
#include <string> 

using namespace std;
using namespace cv;

const int WIDTH = 1600;               //camera resolution       
const int HEIGHT= 1200;
cv::Size imageSize = cv::Size(WIDTH, HEIGHT);
cv::Mat rgbImageL, grayImageL, grayImageLF;
cv::Mat rgbImageR, grayImageR, grayImageRF;
cv::Mat rectifyImageL, rectifyImageR;

cv::Rect validROIL;//after rectification, the image will be cutted. The validROI is the new region after cut. 
cv::Rect validROIR;
cv::Mat mapLx, mapLy, mapRx, mapRy;//mapping list
cv::Mat Rl, Rr, Pl, Pr, Q; //rectification rotation matrix R, projection matrix P and Reprojection Matrix Q.
cv::Mat xyz,xyz1;// coordinate in 3d.
cv::Point origin;//origin from the mouseclick
cv::Rect selection;//define the rectangle
cv::Mat img1p, img2p;
bool selectObject = false;//whether to slect the object

/*********use the calibrated camera parameters from Matlab*************/
cv::Mat cameraMatrixL = (cv::Mat_<double>(3,3) << 
						 1683.34403, 2.69616, 788.48332,
						 0.00000, 1684.73974, 535.82890,
						 0.00000, 0.00000, 1.00000);
cv::Mat distCoeffL = (cv::Mat_<double>(5,1) << -0.23821, 0.15847, -0.00107, 0.00448, 0.00000);

cv::Mat cameraMatrixR = (cv::Mat_<double>(3,3) << 
						 1687.55162, -0.00468, 798.93488,
						 0.00000, 1688.44087, 530.11714,
						 0.00000, 0.00000, 1.00000);
cv::Mat distCoeffR = (cv::Mat_<double>(5,1) << -0.24737, 0.19879, -0.00193, -0.00064, 0.00000);

cv::Mat T = (cv::Mat_<double>(3,1)<< -144.11441, -0.81991, 0.10386);

//cv::Mat rec = (cv::Mat_<double>(3,1)<<-0.00049, -0.00116, -0.00288);
cv::Mat R=(cv::Mat_<double>(3,3) << 
						 1.00000, 0.00451, -0.01247,
						 -0.00451, 1.00000, 0.00313,
						 0.01247, -0.00313, 1.00000);;// rotation matrix
int ElasMatch(cv::Mat leftImage, cv::Mat rightImage);

cv::Mat disp_l, disp_r, disp8u_l, disp8u_r, disp_lf, disp_rf;
/********Read images in folder********/
vector<Mat> read_images_in_folder(cv::String pattern)
{
    vector<cv::String> fn;
    glob(pattern, fn, false);

    vector<Mat> images;
    size_t count = fn.size(); //number of files in images folder
    for (size_t i = 0; i < count; i++)
    {
        images.push_back(imread(fn[i]));
        //imshow("img", imread(fn[i]));
        //waitKey(1000);
    }
    return images;
}

//MouseCallbck settings

static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
	}
	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:   //event selected from the left button
		origin = cv::Point(x, y);
		selection = cv::Rect(x, y, 0, 0);
		selectObject = true;
		std::cout << origin << "in world coordinate is: " << xyz.at<cv::Vec3f>(origin) << std::endl;
		break;
	case cv::EVENT_LBUTTONUP:    //event released from the left button
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}


static void saveXYZ(const char*filename, const cv::Mat& mat)
{
	const double max_z = 8.0e3;
	FILE* fp =fopen(filename, "wt");
	for (int x=0; x<mat.cols;x++)                ///!!!!!!!!!!!!if the data will be used in Matlab, please in this order:first cols, then rows
    {
	    for (int y=0; y<mat.rows;y++)
	    {
		
			cv::Vec3f point = mat.at<cv::Vec3f>(y,x);
			//if (fabs(point[2]-max_z)<FLT_EPSILON || fabs(point[2])>max_z) continue;
			if (fabs(point[2]-max_z)<FLT_EPSILON || fabs(point[2])>max_z)
				//fprintf(fp, "%f\t%f\t%s\n", point[0], point[1], "Inf");
			fprintf(fp, "%s\t%s\t%s\n", "Inf", "Inf", "Inf");
			else
				fprintf(fp, "%f\t%f\t%f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

/******************paint some color for the depth image.********/
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
	//color map
	float max_val = 225.0f;
	float map[8][4] = {{0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
	{0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0}};
    float sum = 0;
	for (int i=0; i<8;i++)
		sum += map[i][3];

	float weights[8];//relative weight
	float cumsum[8];//cumulative weights
	cumsum[0] = 0;
	for( int i = 0; i<7; i++)
	{
		weights[i]=sum/map[i][3];
		cumsum[i+1]= cumsum[i] + map[i][3]/ sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;

	//for all pixels do
	for(int v=0; v<height_; v++)
	{
		for (int u=0; u<width_; u++)
		{
			//get normaliazed value
			float val = std::min(std::max(src.data[v * width_ + u] / max_val, 0.0f),1.0f);

			//find bin
			int i;
			for (i=0; i<7; i++)
				if(val<cumsum[i+1])
					break;

			//compute red/green/blue values
			float   w = 1.0 - (val - cumsum[i]) * weights[i];
            uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
            uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
            uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
            //rgb memeroy constant storage 
            disp.data[v*width_ * 3 + 3 * u + 0] = b;
            disp.data[v*width_ * 3 + 3 * u + 1] = g;
            disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}


int main() {
	/*stereo rectification*/
	//cv::Rodrigues(rec, R);// Rodrigues transformation.
	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
		0, imageSize, &validROIL, &validROIR );

	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);

    String left = "C://Users//m8avhru//Documents//MATLAB//Projects//Stereo-Odometry-SOFT-master//data//Frame_L//*.jpg";
	vector<Mat> images_left = read_images_in_folder(left);
	String right = "C://Users//m8avhru//Documents//MATLAB//Projects//Stereo-Odometry-SOFT-master//data//Frame_R//*.jpg";
	vector<Mat> images_right = read_images_in_folder(right);
	//Mat left = imread("C://Users//m8avhru//Desktop//Calibration//FrameSelectedR//RightFrame275.jpg",IMREAD_GRAYSCALE);////////////////////////////////////************
	//Mat right = imread("C://Users//m8avhru//Desktop//Calibration//FrameSelectedL//LeftFrame275.jpg", IMREAD_GRAYSCALE);//////////////////////////////////*************
	for (int i = 0; i<images_left.size(); i++)
	{
		Mat rgbImageL = images_left[i];
		Mat rgbImageR = images_right[i];
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

		//cv::imshow("l10", images_left[10]);
		//cv::imshow("r10", images_right[10]);

		cv::remap(grayImageL, rectifyImageL, mapLx, mapLy, CV_INTER_LINEAR);
	    cv::remap(grayImageR, rectifyImageR, mapRx, mapRy, CV_INTER_LINEAR);



	    //cv::namedWindow("disparity", CV_WINDOW_NORMAL);////

	    //cv::setMouseCallback("disparity", onMouse, 0); ////

	    cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, 255, 255, IPL_BORDER_REPLICATE);
	    cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, 255, 255, IPL_BORDER_REPLICATE);
	    ElasMatch(img1p, img2p);

	    cv::reprojectImageTo3D(disp_lf, xyz, Q, false);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
		auto s = std::to_string(i);
		
		string extension = ".csv";
		string extension_disp= ".png";
		string name;
		string name_disp;
		name = s + extension;
		name_disp = s + extension_disp;
		//vector<char> v(name.begin(), name.end());
		char *ca = (char*)name.c_str();
		char *nameofdisp = (char*)name_disp.c_str();
		cv::imwrite(nameofdisp, disp_lf);

	    saveXYZ(ca, xyz);///////////////////////////////////////////////////////////////////////////////////////////
	}
	
	cv::waitKey(0);
	return 0;
}

int ElasMatch(cv::Mat leftImage, cv::Mat rightImage)
{
	
	double minVal; double maxVal; 

	// generate disparity image using LIBELAS
	int bd = 0;
	const int32_t dims[3] = { leftImage.cols,leftImage.rows,leftImage.cols };
	cv::Mat leftdpf = cv::Mat::zeros(cv::Size(leftImage.cols, leftImage.rows), CV_32F);
	cv::Mat rightdpf = cv::Mat::zeros(cv::Size(leftImage.cols, leftImage.rows), CV_32F);
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	double matching_time = (double)cv::getTickCount();
	elas.process(leftImage.data, rightImage.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
	matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
	std::cout<< "matching_time="<<matching_time<<std::endl;
	cv::Mat(leftdpf(cv::Rect(bd, 0, leftImage.cols, leftImage.rows))).copyTo(disp_l);
	cv::Mat(rightdpf(cv::Rect(bd, 0, rightImage.cols, rightImage.rows))).copyTo(disp_r);

	disp_lf = disp_l.colRange(255, img1p.cols-255);
	
	disp_rf = disp_r.colRange(255, img2p.cols- 255);
	//-- Check its extreme values
	cv::minMaxLoc(disp_lf, &minVal, &maxVal);
	cout << "Min disp: Max value" << minVal << maxVal; //numberOfDisparities.= (maxVal - minVal)

	//-- Display it as a CV_8UC1 image
	disp_lf.convertTo(disp8u_l, CV_8U, 255 / (maxVal - minVal));//(numberOfDisparities*16.)

	cv::minMaxLoc(disp_rf, &minVal, &maxVal);
	cout << "Min disp: Max value" << minVal << maxVal; //numberOfDisparities.= (maxVal - minVal)

	//-- Display it as a CV_8UC1 image
	disp_rf.convertTo(disp8u_r, CV_8U, 255 / (maxVal - minVal));//(numberOfDisparities*16.)

	cv::normalize(disp8u_l, disp8u_l, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image
	cv::normalize(disp8u_r, disp8u_r, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image

	
	//xyz = xyz * 16;


	//cv::imshow("Left", leftImage);
	//cv::imshow("Right", rightImage);

	//cv::imshow("Elas_left", disp8u_l);
	//cv::imshow("Elas_right", disp8u_r);
	//cv::imwrite("Elas_left_275.png", disp8u_l);//////////
	//cv::imwrite("Elas_right_275.png", disp8u_r);/////////

	//Mat left_disp = imread("1L_disp.pgm");
	//imshow("left",left_disp);
	//cv::Mat color(disp_lf.size(), CV_8UC3);
	
	//GenerateFalseMap(disp8u_l, color);// to color map.
	//cv::imshow("disparity", color);
	
	//cout << endl << "Over" << endl;
	//cv::waitKey(0);

	return 0;
}




