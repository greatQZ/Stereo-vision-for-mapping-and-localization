#include <opencv2/opencv.hpp>
#include <iostream> 
#include <stdio.h>
#include <math.h>
#include <string> 


using namespace std;
using namespace cv;

const int WIDTH=1600;               //camera resolution       
const int HEIGHT=1200;
cv::Size imageSize = cv::Size(WIDTH, HEIGHT);

cv::Mat rgbImageL, grayImageL, grayImageLF;
cv::Mat rgbImageR, grayImageR, grayImageRF;
cv::Mat rectifyImageL, rectifyImageR;

cv::Rect validROIL;//after rectification, the image will be cutted. The validROI is the new region after cut. 
cv::Rect validROIR;

cv::Mat mapLx, mapLy, mapRx, mapRy;//mapping list
cv::Mat Rl, Rr, Pl, Pr, Q; //rectification rotation matrix R, projection matrix P and Reprojection Matrix Q.
cv::Mat xyz,xyz1;// coordinate in 3d.
cv::Mat img1p, img2p;
cv::Point origin;//origin from the mouseclick
cv::Rect selection;//define the rectangle
bool selectObject = false;//whether to slect the object

int blockSize = 0, uniquenessRatio =0, SpeckleWindowSize=0, SpeckleRange=0; 
int numDisparities = ((imageSize.width/8) + 15) & -16;
cv::Mat displ,dispr,dispfl,dispfr,disp32_l,disp32_r;
cv::Mat lchk =  cv::Mat::zeros( HEIGHT, WIDTH, CV_8U );
cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,16,3);
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
static void print_help()
{
	std::cout<<"Demo stereo matching converting L and R images into disparity and point clouds"<<std::endl;
	std::cout<<"Usage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way]"<<std::endl;
}
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

/*****************left right check***************************************/
void leftrightcheck(const cv::Mat& DSP_left, const cv::Mat& DSP_right, cv::Mat& output)
{
	IplImage* leftImg= cvCreateImage(cvSize(WIDTH,HEIGHT), 8, 1);
	IplImage* rightImg=cvCreateImage(cvSize(WIDTH,HEIGHT), 8, 1);
	//int scale;
	int w,h,nchs,step,depth;

	leftImg->imageData=(char*)DSP_left.data;
	cvShowImage("leftImg", leftImg); 
	rightImg->imageData=(char*)DSP_right.data;
	cvShowImage("rightImg", rightImg); 
	//scale=16;

	depth=leftImg->depth;
	w=leftImg->width;
	h=leftImg->height;
	nchs=leftImg->nChannels;
	//std::cout<<nchs<<std::endl;
	step=leftImg->widthStep/sizeof(uchar);
	


	IplImage * result=cvCreateImage(cvSize(w,h),8,1);
	int x,y,zeroNUm;
	
	int p_d_1,p_d_2;
	for(x=0;x<w;x++)
		for(y=0;y<h;y++)	
		{
			p_d_1 = (uchar)leftImg->imageData[y*step+x*nchs];
			//p_d_1 = p_d_1/16;
			/*
			if(x-p_d_1<0)
			{
				result->imageData[y*step+x*nchs]=0;
				break;
			}*/


			p_d_2 = (uchar)rightImg->imageData[y*step+(x-p_d_1)*nchs];
			//p_d_2 = p_d_2/16;
			
			if (abs(p_d_1-p_d_2)>1)
			{
				result->imageData[y*step+x*nchs]=0;
			}
			else		
				result->imageData[y*step+x*nchs]=p_d_1;
		}

	zeroNUm = 0;
	for(x=0;x<w;x++)
		for(y=0;y<h;y++)	
		{
			int t = (uchar)result->imageData[y*step+x*nchs];
			if (t==0)
				zeroNUm++;
		}
	printf("\nzeroNum:%d",zeroNUm);
/*
	for(x=0;x<w;x++)
		for(y=0;y<h;y++)	
		{
			int tem = (uchar)result->imageData[y*step+x*nchs];
			if(tem==0)
			{
				int lp,rp;
				lp = rp = 0;
				
				int lx,rx;
				lx = x;
				rx = x;
				if(lx-1<0)
					lp= (uchar)result->imageData[y*step+lx*nchs];
				while((lp==0)&&( lx-1 >= 0 ))
					lp = (uchar)result->imageData[y*step+(--lx)*nchs];

				if(rx+1>=w)
					rp = (uchar)result->imageData[y*step+rx*nchs];
				while((rp==0)&&(rx+1<w))
					rp = (uchar)result->imageData[y*step+(++rx)*nchs];
				//result->imageData[y*step+x*nchs]=max(lp,rp);
				result->imageData[y*step+x*nchs]=(lp+rp)/2;
			}
		}
	zeroNUm = 0;
	for(x=0;x<w;x++)
		for(y=0;y<h;y++)	
		{
			//result->imageData[y*step+x*nchs] *=16;
			result->imageData[y*step+x*nchs];
			int t = (uchar)result->imageData[y*step+x*nchs];
			if (t==0)
			{
				zeroNUm++;
			}
		}
*/
	cvSmooth(result,result,CV_MEDIAN,3,0,0);

	output.data = (uchar*)result->imageData;
}



/********disp to depth*********/
/*void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
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
				depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]) );
            }
        }
    }
    else
    {
        std::cout << "please confirm dispImg's type!" << std::endl;
        cv::waitKey(0);
    }
}*/


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


/***********SGBM*************/
void stereo_match(cv::Mat leftImage, cv::Mat rightImage)
{
	sgbm->setPreFilterCap(63);
	//int sgbmWinSize= 2*blockSize+3;          //can be set from practice 3-11
	int sgbmWinSize= 11; 

	//std::cout<<numDisparities<<std::endl;
	//int UniquenessRatio = 5*uniquenessRatio+5;     //can be set from practice 5-15
	int UniquenessRatio = 9;
	sgbm->setBlockSize(sgbmWinSize);
	int cn = grayImageL.channels();

	sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numDisparities);
	sgbm->setUniquenessRatio(UniquenessRatio);
	sgbm->setSpeckleWindowSize(300); //set to detect the number of pixels in the connected region 
	sgbm->setSpeckleRange(2); //set a condition to decide whether the two points are in the same region  
	sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_HH);//use MODE_HH for more precise matching from 8 directions instead of 5 directions in MODE_SGBM, but slower.

	/*compute left disparity*/
	double matching_time_L = (double)cv::getTickCount();
	sgbm->compute(leftImage, rightImage, displ);
	matching_time_L = ((double)cv::getTickCount() - matching_time_L)/cv::getTickFrequency();
	std::cout<< "matching_time_L="<<matching_time_L<<std::endl;


	/*compute right disparity*/
	
	/*sgbm->setMinDisparity(-numDisparities);
    sgbm->setNumDisparities(numDisparities);
	double matching_time_R = (double)cv::getTickCount();
    sgbm->compute(rightImage, leftImage, dispr);
	matching_time_R = ((double)cv::getTickCount() - matching_time_R)/cv::getTickFrequency();
	std::cout<< "matching_time_R="<<matching_time_R<<std::endl;*/
	
	//dispr=abs(dispr);
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_dispr.jpg", dispr);


	//cut the black range
	dispfl = displ.colRange(numDisparities, img1p.cols-numDisparities);
	
	/*dispfr = dispr.colRange(numDisparities, img2p.cols- numDisparities);*/

	
	/*double min, max;
    cv::minMaxLoc(dispfr, &min, &max);
	std::cout<<min<< " " << max<< std::endl;
	dispfr= abs(dispfr);
	for(int x=0; x<dispfr.rows;x++)
		for(int y=0; y<dispfr.cols; y++)
		{
			if((int)dispfr.at<short>(x, y) > (numDisparities-1)*16)//filter all the unreliable value to 0.
				dispfr.at<short>(x, y)= 0;
		}
	*/	
	dispfl.convertTo(disp32_l, CV_8UC1, 1.0/16); 
	/*dispfr.convertTo(disp32_r, CV_8UC1, 1.0/16);*/ 
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_indoor_dispfl_322.png", disp32_l);////////////////////****************
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_indoor_dispfr_322.png", disp32_r);////////////////////////*****************

	
	//dispfl.convertTo(disp8_l, CV_8U, 225 / (numDisparities*16.));
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_indoor_disp8_l.png", disp8_l);
	//dispfr.convertTo(disp8_r, CV_8U, 225 / (numDisparities*16.));
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_indoor_disp8_r.png", disp8_r);


	//leftrightcheck
	/*double left_right_check_time = (double)cv::getTickCount();
	leftrightcheck(disp32_l, disp32_r, lchk);
	left_right_check_time = ((double)cv::getTickCount() - left_right_check_time)/cv::getTickFrequency();
	std::cout<< "left_right_check_time="<<left_right_check_time<<std::endl;*/
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//disp_indoor_chk_322.png", lchk);///////////////*****************************
	cv::reprojectImageTo3D(disp32_l, xyz, Q, false);// when calculate the distance practically, the X/W, Y/W, Z/W from reprojection must times 16
	
	
	//cv::reprojectImageTo3D(disp32_r, xyz1, Q, true);
	//xyz = xyz * 16;
	//lchk.convertTo(disp8_l, CV_8U, 225 / (numDisparities*16.));
	//cv::normalize(lchk, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cv::Mat color(lchk.size(), CV_8UC3);
	
	//GenerateFalseMap(lchk, color);// to color map.
    //cv::imshow("disparity", color);

	//saveXYZ("xyz322.csv", xyz);
	//saveXYZ("xyz1.xls", xyz1);

	//saveDisp("dispmap.txt", dispf);
	/*cv::Mat dispMap, depthMap,depth_Map;///////////////////////////////////
	dispf.convertTo(dispMap, CV_8U);////////////////////////////////
	dispf.convertTo(depthMap, CV_16UC1);
	disp2Depth(disp8, depthMap, cameraMatrixL);/////////////////////////////
    cv::imwrite("C://Users//m8avhru//Desktop//Calibration//sgbm_output_depth.png", depthMap);*/

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




int main(int argc, char** argv) 
{
	/*stereo rectification*/
	//cv::Rodrigues(rec, R);// Rodrigues transformation.
	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
		0, imageSize, &validROIL, &validROIR );

	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);


	//read images
    String left = "C://Users//m8avhru//Documents//MATLAB//Projects//Stereo-Odometry-SOFT-master//data//Frame_L//*.jpg";
	vector<Mat> images_left = read_images_in_folder(left);
	String right = "C://Users//m8avhru//Documents//MATLAB//Projects//Stereo-Odometry-SOFT-master//data//Frame_R//*.jpg";
	vector<Mat> images_right = read_images_in_folder(right);
	//rgbImageL = cv::imread("C://Users//m8avhru//Desktop//Calibration//FrameSelectedR//RightFrame322.jpg", CV_LOAD_IMAGE_COLOR); ///////////CV_LOAD_IMAGE_COLOR//*****
	//cv::cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
	//if(rgbImageL.empty())
      // std::cout << "failed to open Frame_L.jpg" << std::endl;
    //else
       //std::cout << "Frame_1_L.jpg loaded OK" << std::endl;
	
	//rgbImageR = cv::imread("C://Users//m8avhru//Desktop//Calibration//FrameSelectedL//LeftFrame322.jpg", -1);////////////////////////*************************
	//cv::cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
	//if(rgbImageR.empty())
       //std::cout << "failed to open Frame_R.jpg" << std::endl;
    //else
       //std::cout << "Frame_1_R.jpg loaded OK" << std::endl;
	
	/*cv::imshow("ImageL Before Rectify", grayImageL);
	cv::imshow("ImageR Before Rectify", grayImageR);*/
	for (int i = 0; i<images_left.size(); i++)
	{
		Mat rgbImageL = images_left[i];
		Mat rgbImageR = images_right[i];
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
		cv::remap(grayImageL, rectifyImageL, mapLx, mapLy, CV_INTER_LINEAR);
	    cv::remap(grayImageR, rectifyImageR, mapRx, mapRy, CV_INTER_LINEAR);
	


	/*Pre-Processing Filter*/
	//grayImageLF = grayImageL.clone();
	//grayImageRF = grayImageR.clone();
		
	cv::copyMakeBorder(rectifyImageL, img1p, 0, 0, numDisparities, numDisparities, IPL_BORDER_REPLICATE);
	cv::copyMakeBorder(rectifyImageR, img2p, 0, 0, numDisparities, numDisparities, IPL_BORDER_REPLICATE);
    stereo_match(img1p,img2p);

	auto s = std::to_string(i);

    string extension = ".csv";
	string extension_disp= ".png";
	string name;
	string name_disp;
	//string name_l;
	//string l = "l_";
	//string r = "r_";
	//string name_r;
	name = s + extension;
	name_disp = s + extension_disp;
	//name_l = l+s+extension_disp;
	//name_r = r+s+extension_disp;
	//vector<char> v(name.begin(), name.end());
	char *ca = (char*)name.c_str();
	char *nameofdisp = (char*)name_disp.c_str();
	cv::imwrite(nameofdisp, disp32_l);
	//cv::imwrite(name_l, rectifyImageL);
	//cv::imwrite(name_r, rectifyImageR);
	saveXYZ(ca, xyz);///////////////////////////////////////////////////////////////////////////////////////////
	}


	//cv::bilateralFilter(grayImageL, grayImageLF, 9, 100, 100);
	//cv::bilateralFilter(grayImageR, grayImageRF, 9, 100, 100);
	//cv::medianBlur(grayImageL, grayImageLF, 7);
	//cv::medianBlur(grayImageR, grayImageRF, 7);
	//cv::GaussianBlur(grayImageL, grayImageLF, cv::Size(5,5), 0, 0);
	//cv::GaussianBlur(grayImageR, grayImageRF, cv::Size(5,5), 0, 0);

	//cv::imshow("Filteredl", grayImageLF);
	//cv::imshow("Filteredr", grayImageRF);
    /*cv::Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //pseudo-color map
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);*/
    
		
    //cv::imwrite("C://Users//m8avhru//Desktop//Calibration//rectifyImageL_Indoor.jpg", rectifyImageL);
	//cv::imwrite("C://Users//m8avhru//Desktop//Calibration//rectifyImageR_Indoor.jpg", rectifyImageR);

	//in the same image
    /*cv::Mat canvas;
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

	*/
	//cv::namedWindow("disparity", CV_WINDOW_NORMAL);
	//cv::createTrackbar("BlockSize:\n", "disparity",&blockSize, 4, stereo_match);
	//cv::createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 3, stereo_match);
	//cv::createTrackbar("SpeckleWindowSize:\n", "disparity", &SpeckleWindowSize, 400, stereo_match);
	//cv::createTrackbar("SpeckleRange:\n", "disparity", &SpeckleRange, 30, stereo_match);


	//Mousecallback function
	//cv::setMouseCallback("disparity", onMouse, 0); 




	cv::waitKey(0);

	return 0;



	
}  
