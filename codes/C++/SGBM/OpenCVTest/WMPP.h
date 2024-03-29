#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>
#include<bitset>
using namespace std;
using namespace cv;
#pragma  once
#define DOUBLE_MAX 1e10
#define MED_SZ 19
#define SIG_CLR 0.1
#define SIG_DIS 9
/*class PPMethod
{
public:
	PPMethod(void) {}
	virtual ~PPMethod(void) {}
public:
	virtual void postProcess( const Mat& lImg, const Mat& rImg, const int maxDis, const int disSc,
		Mat& lDis, Mat& rDis, Mat& lSeg, Mat& lChk ) = 0;
};*/

//
// Weight-Median Post-processing
//
class WMPP 
{
public:
	WMPP(void) 
	{
		printf( "\n\t\tWeight-Median Post-processing" );
	}
	~WMPP(void) {}
public:
	void postProcess( const Mat& lImg, const Mat& rImg, const int maxDis, Mat& lDis, Mat& rDis, Mat& lSeg, Mat& lChk );
};
