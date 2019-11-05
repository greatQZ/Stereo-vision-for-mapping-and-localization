#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int WIDTH=800;               //camera resolution       
const int HEIGHT=600;
Size imageSize = cv::Size(WIDTH, HEIGHT);

Mat rectifyImageL, rectifyImageR;
Mat img_left, img_right, img_disp;
Mat img_left_desc, img_right_desc;
vector< KeyPoint > kpl, kpr;
int w = 0;

Rect validROIL;//after rectification, the image will be cutted. The validROI is the new region after cut. 
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;//mapping list
Mat Rl, Rr, Pl, Pr, Q; //rectification rotation matrix R, projection matrix P and Reprojection Matrix Q.
//Mat xyz;// coordinate in 3d.

//Point origin;//origin from the mouseclick
//Rect selection;//define the rectangle
//bool selectObject = false;//whether to slect the object

/*********use the calibrated camera parameters from Matlab*************/

Mat cameraMatrixL = (cv::Mat_<double>(3,3)<< 884.29555, 0.00000, 404.96950,
  0.00000, 879.56975, 287.43878,
  0.00000, 0.00000, 1.00000);
Mat distCoeffL = (cv::Mat_<double>(5,1)<< -0.20635, 0.10664, 0.00080, -0.00033, 0.00000);

Mat cameraMatrixR = (cv::Mat_<double>(3,3)<<887.63431, 0.00000, 400.14321,
  0.00000, 881.96613, 287.33474,
  0.00000, 0.00000, 1.00000);
Mat distCoeffR = (cv::Mat_<double>(5,1)<< -0.22488, 0.19033, 0.00042, 0.00201, 0.00000);

Mat T = (cv::Mat_<double>(3,1)<< -153.85361, -1.21912, 0.47128);

Mat rec = (cv::Mat_<double>(3,1)<<-0.00049, -0.00116, -0.00288);
Mat R;// rotation matrix




bool inImg(Mat& img, int x, int y) {
  if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
    return true;
}

bool isLeftKeyPoint(int i, int j) {
  int n = kpl.size();
  return (i >= kpl[0].pt.x && i <= kpl[n-1].pt.x
          && j >= kpl[0].pt.y && j <= kpl[n-1].pt.y);
}

bool isRightKeyPoint(int i, int j) {
  int n = kpr.size();
  return (i >= kpr[0].pt.x && i <= kpr[n-1].pt.x
          && j >= kpr[0].pt.y && j <= kpr[n-1].pt.y);
}

long descCost(Point leftpt, Point rightpt, int w) {
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  long cost = 0;
  for (int j = -w; j <= w; j++) {
    for (int k = -w; k <= w; k++) {
      if (!isLeftKeyPoint(leftpt.x+j, leftpt.y+k) || 
          !isRightKeyPoint(rightpt.x+j, rightpt.y+k))
        continue;
      int idxl = (leftpt.x+j-x0l)*(ynl-y0l+1)+(leftpt.y+k-y0l);
      int idxr = (rightpt.x+j-x0r)*(ynr-y0r+1)+(rightpt.y+k-y0r);
      cost += norm(img_left_desc.row(idxl), img_right_desc.row(idxr), CV_L1);
    }
  }
  return cost / ((2*w+1)*(2*w+1));
}

/*double descCostNCC(Point leftpt, Point rightpt, int w) {
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  double costL = 0;
  double costR = 0;
  double cost = 0;
  int idxl0 = (leftpt.x-x0l)*(ynl-y0l+1)+(leftpt.y-y0l);
  int idxr0 = (rightpt.x-x0r)*(ynr-y0r+1)+(rightpt.y-y0r);
  for (int j = -w; j <= w; j++) {
    for (int k = -w; k <= w; k++) {
      if (!isLeftKeyPoint(leftpt.x+j, leftpt.y+k) || 
          !isRightKeyPoint(rightpt.x+j, rightpt.y+k))
        continue;
      int idxl = (leftpt.x+j-x0l)*(ynl-y0l+1)+(leftpt.y+k-y0l);
      int idxr = (rightpt.x+j-x0r)*(ynr-y0r+1)+(rightpt.y+k-y0r);
      double d1 = norm(img_left_desc.row(idxl), img_left_desc.row(idxl0), 
                       CV_L1);
      double d2 = norm(img_right_desc.row(idxr), img_right_desc.row(idxr0), 
                       CV_L1);
      costL += d1*d1;
      costR += d2*d2;
      cost += d1*d2;
    }
  }
  cost /= (sqrt(costL) * sqrt(costR));
  cout << "ncc: " << cost << endl;
  return cost;
}*/

int getCorresPointRight(Point p, int ndisp) {
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    long cost = descCost(p, Point(i,p.y), w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (minCost == 0)
    return p.x;
  return chosen_i;
  /*
  double corr = -10;
  int chosen_i = 0;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    double cost = descCostNCC(p, Point(i,p.y), w);
    if (cost > corr) {
      corr = cost;
      chosen_i = i;
    }
  }
  cout << "corr: " << corr << endl;
  return chosen_i;
  */
}

int getCorresPointLeft(Point p, int ndisp) {
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x; i <= p.x+ndisp; i++) {
    long cost = descCost(Point(i,p.y), p, w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (minCost == 0)
    return p.x;
  return chosen_i;
}


void computeDisparityMapORB(int ndisp) {
    img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
    for (int i = ndisp+1; i < img_left.cols; i++) {
      for (int j = 0; j < img_left.rows; j++) {
        cout << i << ", " << j << endl;
        if (!isLeftKeyPoint(i,j))
          continue;
        int right_i = getCorresPointRight(Point(i,j), ndisp);
      // left-right check
      /*
      int left_i = getCorresPointLeft(Point(right_i,j), ndisp);
      if (abs(left_i-i) > 4)
        continue;
      */
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity;
    }
  }
}

void cacheDescriptorVals() {
  Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
  //BriefDescriptorExtractor extractor;
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      kpl.push_back(KeyPoint(i,j,1));
      kpr.push_back(KeyPoint(i,j,1));
    }
  }
  descriptorExtractor->compute(img_left, kpl, img_left_desc);
  descriptorExtractor->compute(img_right, kpr, img_right_desc);
}

/*void preprocess(Mat& img) {
  Mat dst;
  bilateralFilter(img, dst, 10, 15, 15);
  img = dst.clone();
}*/



/*static void onMouse(int event, int x, int y, int, void*)
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
}*/

int main(int argc, char const *argv[])
{
  /*stereo rectification*/
	/*Rodrigues(rec,R);// Rodrigues transformation.
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
		0, imageSize, &validROIL, &validROIR );

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);
	*/
    img_left = imread("C://Users//m8avhru//Desktop//Calibration//frameL.png", 1);
	if(img_left.empty())
		cout << "Failed to open image" << endl;
	else
		cout << "image loaded ok" <<endl;
    img_right = imread("C://Users//m8avhru//Desktop//Calibration//frameR.png", 1);
	if(img_right.empty())
		cout << "Failed to open image" << endl;
	else
		cout << "image loaded ok" <<endl;

	//remap(img_left, rectifyImageL, mapLx, mapLy, CV_INTER_LINEAR);
	//remap(img_right, rectifyImageR, mapRx, mapRy, CV_INTER_LINEAR);

	//in the same image
    //cv::Mat canvas;
	/*double sf;   
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
    
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
    
	//canvas.create(h, w * 2, CV_8UC3); // care about the canvas

	//left image draw on the canvas

    //cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));                //get a part of canvas
	//cv::resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA); //scale the image size same as canvasPart
	cv::Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //get the cutted region
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));  
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //draw a rectangel
	//std::cout << "Painted ImageL" << std::endl;


	//right image draw on the canvas
	//canvasPart = canvas(cv::Rect(w, 0, w, h));
	//cv::resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR);
	cv::Rect vroiR(cvRound(validROIR.x*sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width*sf), cvRound(validROIR.height*sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	//std::cout << "Painted ImageR" << std::endl;

	//paint the correspondent lines
	//for(int i=0; i<canvas.rows; i+=16)
		line(canvas, cv::Point(0,i), cv::Point(canvas.cols,i), cv::Scalar(0, 255, 0), 1, 8);
	cv::imshow("rectified", canvas);

	//save the rectified images
	cv::imwrite("rectified.jpg", canvas);*/



  //preprocess(img_left);
  //preprocess(img_right);

    cacheDescriptorVals();



  //namedWindow("IMG-LEFT", 1);
  //namedWindow("IMG-RIGHT", 1);

  	//cv::namedWindow("disparity", CV_WINDOW_NORMAL);

	//Mousecallback function
	//cv::setMouseCallback("disparity", onMouse, 0); //Disparity

	computeDisparityMapORB(((imageSize.width / 8) + 15) & -16);

	imwrite("C://Users//m8avhru//Desktop//Calibration//Orb_output.jpg", img_disp);
	imshow("IMG-DISP", img_disp);
	waitKey(0);
    return 0;
}