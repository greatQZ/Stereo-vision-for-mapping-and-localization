#include <opencv2/opencv.hpp>
#include <iostream> 
#include<stdio.h>


const int WIDTH=800;               //camera resolution       
const int HEIGHT=600;
cv::Size imageSize = cv::Size(WIDTH, HEIGHT);

cv::Mat rgbImageL, grayImageL;
cv::Mat rgbImageR, grayImageR;
cv::Mat rectifyImageL, rectifyImageR;

cv::Rect validROIL;//after rectification, the image will be cutted. The validROI is the new region after cut. 
cv::Rect validROIR;

cv::Mat mapLx, mapLy, mapRx, mapRy;//mapping list
cv::Mat Rl, Rr, Pl, Pr, Q; //rectification rotation matrix R, projection matrix P and Reprojection Matrix Q.
cv::Mat xyz;// coordinate in 3d.

cv::Point origin;//origin from the mouseclick
cv::Rect selection;//define the rectangle
bool selectObject = false;//whether to slect the object

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
//cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,16,3);

/*********use the calibrated camera parameters from Matlab*************/

cv::Mat cameraMatrixL = (cv::Mat_<double>(3,3)<< 884.29555, 0.00000, 404.96950,
  0.00000, 879.56975, 287.43878,
  0.00000, 0.00000, 1.00000);
cv::Mat distCoeffL = (cv::Mat_<double>(5,1)<< -0.20635, 0.10664, 0.00080, -0.00033, 0.00000);

cv::Mat cameraMatrixR = (cv::Mat_<double>(3,3)<<887.63431, 0.00000, 400.14321,
  0.00000, 881.96613, 287.33474,
  0.00000, 0.00000, 1.00000);
cv::Mat distCoeffR = (cv::Mat_<double>(5,1)<< -0.22488, 0.19033, 0.00042, 0.00201, 0.00000);

cv::Mat T = (cv::Mat_<double>(3,1)<< -153.85361, -1.21912, 0.47128);

cv::Mat rec = (cv::Mat_<double>(3,1)<<-0.00049, -0.00116, -0.00288);
cv::Mat R;// rotation matrix


/*static void print_help()
{
	std::cout<<"Demo stereo matching converting L and R images into disparity and point clouds"<<std::endl;
	std::cout<<"Usage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way]"<<std::endl;
}*/


static void saveXYZ(const char*filename, const cv::Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp =fopen(filename, "wt");
	for (int y=0; y<mat.rows;y++)
	{
		for (int x=0; x<mat.cols;x++)
		{
			cv::Vec3f point = mat.at<cv::Vec3f>(y,x);
			if (fabs(point[2]-max_z)<FLT_EPSILON || fabs(point[2])>max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}


/******************paint some color for the depth image.********/
/*void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
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

*/



/*****************STERTEO MATCHING*********************/


/*******BM********/

/*void stereo_match(int,void*)
{
    bm->setBlockSize(2*blockSize+5);     //the size of SAD window, 5~21 is well
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);  
    bm->setNumDisparities(numDisparities*16+16);
    bm->setTextureThreshold(10); 
    bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio to prevent false matching
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);
    Mat disp, disp8;
    bm->compute(rectifyImageL, rectifyImageR, disp);// the input images must be gray
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//disparity in form of CV_16S
    reprojectImageTo3D(disp, xyz, Q, true); 
    xyz = xyz * 16;
    imshow("disparity", disp8);
}*/


/***********SGBM*************/
void stereo_match(int, void*)
{
	int NumDisparities = ((imageSize.width / 8) + 15) & -16;
	/*sgbm->setPreFilterCap(63);
	int sgbmWinSize= 5;          //can be set from practice
	//int NumDisparities = 416;    //can be set from practice
	int NumDisparities = ((imageSize.width / 8) + 15) & -16;
	int uniquenessRatio = 10;     //can be set from practice
	sgbm->setBlockSize(sgbmWinSize);
	int cn = rectifyImageL.channels();

	sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(NumDisparities);
	sgbm->setUniquenessRatio(uniquenessRatio);
	sgbm->setSpeckleWindowSize(100); //set to detect the number of pixels in the connected region 
	sgbm->setSpeckleRange(10); //set a condition to decide whether the two points are in the same region  
	sgbm->setDisp12MaxDiff(1);

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);*/



	/****************For SGBM_output************************/
	/*cv::Mat disp,dispf,disp8;
    cv::Mat img1p, img2p;
	disp= cv::imread("C://Users//m8avhru//Desktop//Calibration//sgbm_output.jpg", -1);
	rectifyImageL = ("C://Users//m8avhru//Desktop//Calibration//rectifyImageL.jpg", CV_LOAD_IMAGE_COLOR);
	rectifyImageR = ("C://Users//m8avhru//Desktop//Calibration//rectifyImageR.jpg", CV_LOAD_IMAGE_COLOR);
	cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	
	dispf = disp.colRange(NumDisparities, img2p.cols- NumDisparities);


	dispf.convertTo(disp8, CV_8U, 225 / (NumDisparities*16.));

	cv::reprojectImageTo3D(dispf, xyz, Q, true);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
	xyz = xyz * 16;
	cv::imshow("disparity", disp8);
	saveXYZ("xyz.xls", xyz);
	*/
	

	/****************For output1 or Orb_output.jpg********************************/
	cv::Mat disp8;
	disp8= cv::imread("C://Users//m8avhru//Desktop//Calibration//Elas_left.png", -1);
	//cut the black range
	/*cv::Mat img1p, img2p, dispf;
	cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, ((imageSize.width / 8) + 15) & -16, 0, IPL_BORDER_REPLICATE);
	cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, ((imageSize.width / 8) + 15) & -16, 0, IPL_BORDER_REPLICATE);*/
	//dispf = disp.colRange(((imageSize.width / 8) + 15) & -16, img2p.cols- ((imageSize.width / 8) + 15) & -16);
	//disp.convertTo(disp8, CV_8U, 255 / (((imageSize.width / 8) + 15)*16.));

	cv::reprojectImageTo3D(disp8, xyz, Q, true);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
	//cv::imshow("disparity", disp8);
//	cv::Mat color(dispf.size(), CV_8UC3);

//	GenerateFalseMap(disp8, color);// to color map.
    cv::imshow("disparity", disp8);

	saveXYZ("xyz.xls", xyz); 
	
    /*******************For Elas_left.png***********************/
	/*cv::Mat disp,dispf,disp8;
    cv::Mat img1p, img2p;
	cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	disp8= cv::imread("C://Users//m8avhru//Desktop//Calibration//Elas_left.png", -1);
	dispf = disp8.colRange(NumDisparities, img2p.cols- NumDisparities);


	cv::reprojectImageTo3D(dispf, xyz, Q, true);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
	xyz = xyz * 16;
	cv::imshow("disparity", disp8);
	saveXYZ("xyz.xls", xyz);
    */
}



/****************MouseCallbck settings*******************/

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




int main(int argc, char** argv) 
{
	/*stereo rectification*/
	cv::Rodrigues(rec,R);// Rodrigues transformation.
	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
		0, imageSize, &validROIL, &validROIR );

	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);


/*	//read images
	rgbImageL = cv::imread("C://Users//m8avhru//Documents//Visual Studio 2012//Projects//OpenCVTest//Frame_1_old_L.jpg", CV_LOAD_IMAGE_COLOR); ///////////CV_LOAD_IMAGE_COLOR
	//cv::cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
	if(rgbImageL.empty())
       std::cout << "failed to open Frame_L.jpg" << std::endl;
    else
       std::cout << "Frame_1_L.jpg loaded OK" << std::endl;

	rgbImageR = cv::imread("C://Users//m8avhru//Documents//Visual Studio 2012//Projects//OpenCVTest//Frame_1_old_R.jpg", -1);
	//cv::cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
	if(rgbImageR.empty())
       std::cout << "failed to open Frame_R.jpg" << std::endl;
    else
       std::cout << "Frame_1_R.jpg loaded OK" << std::endl;
*/
	/*cv::imshow("ImageL Before Rectify", grayImageL);
	cv::imshow("ImageR Before Rectify", grayImageR);*/


/*	cv::remap(rgbImageL, rectifyImageL, mapLx, mapLy, CV_INTER_LINEAR);
	cv::remap(rgbImageR, rectifyImageR, mapRx, mapRy, CV_INTER_LINEAR);
*/
	

	//display the rectification

    /*cv::Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //pseudo-color map
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);*/
    
	//in the same image
 /*   cv::Mat canvas;
	double sf;   
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
    
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
    
	canvas.create(h, w * 2, CV_8UC3); // care about the canvas

	//left image draw on the canava

    cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));                //get a part of canvas
	cv::resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA); //scale the image size same as canvasPart
	cv::Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //get the cutted region
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));  
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //draw a rectangel
	std::cout << "Painted ImageL" << std::endl;


	//right image draw on the canvas
	canvasPart = canvas(cv::Rect(w, 0, w, h));
	cv::resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR);
	cv::Rect vroiR(cvRound(validROIR.x*sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width*sf), cvRound(validROIR.height*sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	std::cout << "Painted ImageR" << std::endl;

	//paint the correspondent lines
	for(int i=0; i<canvas.rows; i+=16)
		line(canvas, cv::Point(0,i), cv::Point(canvas.cols,i), cv::Scalar(0, 255, 0), 1, 8);
	cv::imshow("rectified", canvas);

	//save the rectified images
	cv::imwrite("rectified.jpg", canvas);





*/

	//stereo matching using SGBM

	cv::namedWindow("disparity", CV_WINDOW_NORMAL);

	//Mousecallback function
	cv::setMouseCallback("disparity", onMouse, 0); //Disparity

	/*******stereo matching using BM
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // build SAD window Trackbar 
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // build uniqueness percentage window of disparity Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // build disparity window Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //setMouseCallback
    setMouseCallback("disparity", onMouse, 0);*/


	stereo_match(0,0);




	cv::waitKey(0);
	return 0;



	
}  
