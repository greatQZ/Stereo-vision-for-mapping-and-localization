#include <opencv2/opencv.hpp>
#include <iostream> 
#include <stdio.h>


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
cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,16,3);

/*********use the calibrated camera parameters from Matlab*************/

cv::Mat cameraMatrixL = (cv::Mat_<double>(3,3) << 
						 884.29555, 0.00000, 404.96950,
						 0.00000, 879.56975, 287.43878,
						 0.00000, 0.00000, 1.00000);
cv::Mat distCoeffL = (cv::Mat_<double>(5,1) << -0.20635, 0.10664, 0.00080, -0.00033, 0.00000);

cv::Mat cameraMatrixR = (cv::Mat_<double>(3,3) << 
						 887.63431, 0.00000, 400.14321,
						 0.00000, 881.96613, 287.33474,
						 0.00000, 0.00000, 1.00000);
cv::Mat distCoeffR = (cv::Mat_<double>(5,1) << -0.22488, 0.19033, 0.00042, 0.00201, 0.00000);

cv::Mat T = (cv::Mat_<double>(3,1)<< -153.85361, -1.21912, 0.47128);

cv::Mat rec = (cv::Mat_<double>(3,1)<<-0.00049, -0.00116, -0.00288);
cv::Mat R;// rotation matrix


static void print_help()
{
	std::cout<<"Demo stereo matching converting L and R images into disparity and point clouds"<<std::endl;
	std::cout<<"Usage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way]"<<std::endl;
}

/********disp to depth*********/
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
{
    int type = dispMap.type();
    float fx = K.at<double>(0, 0);
    float fy = K.at<double>(1, 1);
    float cx = K.at<double>(0, 2);
    float cy = K.at<double>(1, 2);
    float baseline = 153.85361;
    if (type == CV_8U)
    {
        const float PI = 3.14159265358;
		int height = dispMap.rows;
        int width = dispMap.cols;
        //uchar* dispData = (uchar*)dispMap.data;
        //ushort* depthData = (ushort*)depthMap.data;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                //int id = i*width + j;

                if (!((float)dispMap.at<uchar>(i, j)))  continue; //prevent to divide 0
				//std::cout<< (float)dispMap.at<uchar>(i, j) <<std::endl;
				//std::cout<< "here" <<std::endl;
				depthMap.at<ushort>(i, j) = ushort( (float)fx *baseline / ((float)(dispMap.at<uchar>(i, j)+4.82629 )));
                //depthMap.at<ushort>(i, j) = ushort( (float)fx *baseline / ((float)dispMap.at<uchar>(i, j)) );///problem


				/*int id = i*width + j;
				if (!dispData[id])  continue;  //prevent to divide 0
				std::cout << "here" << std::endl;
				std::cout<< depthMap.data <<std::endl;
				std::cout<< depthData[id] <<std::endl;
				std::cout << "here" << std::endl;
				depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]) );*/
            }
        }
    }
    else
    {
        std::cout << "please confirm dispImg's type!" << std::endl;
        cv::waitKey(0);
    }
}


static void saveXYZ(const char*filename, const cv::Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp =fopen(filename, "wt");
	for (int x=0; x<mat.cols;x++)                ///!!!!!!!!!!!!if the data will be used in Matlab, please in this order:first cols, then rows
    {
	    for (int y=0; y<mat.rows;y++)
	    {
		
			cv::Vec3f point = mat.at<cv::Vec3f>(y,x);
			//if (fabs(point[2]-max_z)<FLT_EPSILON || fabs(point[2])>max_z) continue;
			if (fabs(point[2]-max_z)<FLT_EPSILON || fabs(point[2])>max_z)
				fprintf(fp, "%f\t%f\t%s\n", point[0], point[1], "Inf");
			else
				fprintf(fp, "%f\t%f\t%f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

/*void saveDisp(const char* filename, const cv::Mat& mat)		
{
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "%02d\n", mat.rows);
	fprintf(fp, "%02d\n", mat.cols);
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			int disp = (int)mat.at<float>(y, x);	// disparity matrix is CV_16S, so use short to read
			fprintf(fp, "%d\n", disp);			// if disparity matrix is CV_32F, then use float to read
		}
		//fprintf(fp, "\n");
	}
	fclose(fp);
}*/


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
	sgbm->setPreFilterCap(63);
	int sgbmWinSize= 9;          //can be set from practice 3-11
	//int NumDisparities = 416;    //can be set from practice
	//int NumDisparities =  16;
	numDisparities = numDisparities > 0 ? numDisparities : ((imageSize.width/8) + 15) & -16;
	int uniquenessRatio = 10;     //can be set from practice 5-15
	sgbm->setBlockSize(sgbmWinSize);
	int cn = grayImageL.channels();

	sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numDisparities);
	sgbm->setUniquenessRatio(uniquenessRatio);
	sgbm->setSpeckleWindowSize(100); //set to detect the number of pixels in the connected region 
	sgbm->setSpeckleRange(32); //set a condition to decide whether the two points are in the same region  
	sgbm->setDisp12MaxDiff(1);

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);// or use MODE_HH for more precise matching but slower.
	cv::Mat img1p, img2p;
	cv::copyMakeBorder(grayImageL, img1p, 0, 0, numDisparities, 0, IPL_BORDER_REPLICATE);
	cv::copyMakeBorder(grayImageR, img2p, 0, 0, numDisparities, 0, IPL_BORDER_REPLICATE);

	cv::Mat disp,dispf,disp8;
	sgbm->compute(img1p, img2p, disp);
    //disp.convertTo(disp, CV_8U, 255 / (NumDisparities*16.)); 
	cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_disp.jpg", disp);

	//cut the black range

	dispf = disp.colRange(numDisparities, img1p.cols);

	cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_dispf.jpg", dispf);

	dispf.convertTo(disp8, CV_8U, 225 / (numDisparities*16.));

	//disp.convertTo(disp8, CV_8U, 225 / (NumDisparities*16.));

	//cv::reprojectImageTo3D(dispf, xyz, Q, true);
	cv::reprojectImageTo3D(dispf, xyz, Q, true);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
	xyz = xyz * 16;
	
	cv::Mat color(dispf.size(), CV_8UC3);
	cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_disp8.jpg", disp8);
	GenerateFalseMap(disp8, color);// to color map.
    cv::imshow("disparity", color);

	saveXYZ("xyz.xls", xyz);
	//saveDisp("dispmap.txt", dispf);
	cv::Mat dispMap, depthMap,depth_Map;///////////////////////////////////
	//disp_Map = cv::imread("C://Users//m8avhru//Desktop//Calibration//sgbm_output_disp.jpg", 0); ///////////////////////////
	dispf.convertTo(dispMap, CV_8U);////////////////////////////////
	dispf.convertTo(depthMap, CV_16UC1);
	disp2Depth(disp8, depthMap, cameraMatrixL);/////////////////////////////

	cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_depth.png", depthMap);///////////////////////////////

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
	cv::Rodrigues(rec, R);// Rodrigues transformation.
	std::cout << R;
	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
		0, imageSize, &validROIL, &validROIR );

	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);


	//read images
	rgbImageL = cv::imread("C://Users//m8avhru//Desktop//Calibration//ImageL_new.jpg", CV_LOAD_IMAGE_COLOR); ///////////CV_LOAD_IMAGE_COLOR
	//cv::cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
	if(rgbImageL.empty())
       std::cout << "failed to open Frame_L.jpg" << std::endl;
    else
       std::cout << "Frame_1_L.jpg loaded OK" << std::endl;
	
	rgbImageR = cv::imread("C://Users//m8avhru//Desktop//Calibration//ImageR_new.jpg", -1);
	//cv::cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
	if(rgbImageR.empty())
       std::cout << "failed to open Frame_R.jpg" << std::endl;
    else
       std::cout << "Frame_1_R.jpg loaded OK" << std::endl;
	
	/*cv::imshow("ImageL Before Rectify", grayImageL);
	cv::imshow("ImageR Before Rectify", grayImageR);*/


	cv::remap(rgbImageL, rectifyImageL, mapLx, mapLy, CV_INTER_LINEAR);
	cv::remap(rgbImageR, rectifyImageR, mapRx, mapRy, CV_INTER_LINEAR);
	cv::cvtColor(rectifyImageL, grayImageL, CV_BGR2GRAY);
	cv::cvtColor(rectifyImageR, grayImageR, CV_BGR2GRAY);
	//grayImageL= rectifyImageL;
    //grayImageR= rectifyImageR;
	//display the rectification

    /*cv::Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //pseudo-color map
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);*/
    
		
    cv::imwrite("C://Users//m8avhru//Desktop//Calibration//rectifyImageL.jpg", rectifyImageL);
	cv::imwrite("C://Users//m8avhru//Desktop//Calibration//rectifyImageR.jpg", rectifyImageR);

	//in the same image
    cv::Mat canvas;
	double sf;   
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
    
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
    
	canvas.create(h, w * 2, CV_8UC3); // care about the canava

	//left image draw on the canvas

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







	//stereo matching using SGBM

	cv::namedWindow("disparity", CV_WINDOW_NORMAL);
    //set Trackbars
	//cv::createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
	//cv::createTrackbar("UniquenessRatio", "disparity", &uniquenessRatio, 50, stereo_match);
	//cv::createTrackbar("NumDisparities", "disparity", &numDisparities, 16, stereo_match);


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
