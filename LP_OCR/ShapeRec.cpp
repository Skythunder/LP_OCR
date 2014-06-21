#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>
#include <tesseract/baseapi.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;
using namespace  tesseract;

Mat hog(Mat image)
{
	HOGDescriptor hog(cvSize(9,9),cvSize(9,9),cvSize(9,9),cvSize(9,9),64);
	vector<float> ders;
	vector<Point>locs;
	Mat grey;
	cvtColor(image,grey,CV_BGR2GRAY);
	hog.compute(grey,ders,Size(32,32),Size(0,0),locs);
	Mat Hogfeat=Mat::zeros(55000,1,CV_32FC1);
	//Hogfeat.create(ders.size(),1,CV_32FC1);
	for(int i=0;i<ders.size();i++)
	{
		if(i>55000)break;
		Hogfeat.at<float>(i,0)=ders.at(i);

	}
	cout<<ders.size()<<endl;
	return Hogfeat.t();
}

Mat getHist(Mat image,int nbins=256)
{
	Mat hsvimg;
	cvtColor(image,hsvimg,CV_BGR2HSV);
	vector<Mat> hsv_planes,bgr_planes;
	split( hsvimg, hsv_planes );
	split( image, bgr_planes );
	int histSize = nbins;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat h_hist, s_hist, v_hist,r_hist,g_hist,b_hist;
	calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
	normalize(h_hist, h_hist, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize(s_hist, s_hist, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize(v_hist, v_hist, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize(b_hist, h_hist, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, s_hist, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, v_hist, 0, 1, NORM_MINMAX, -1, Mat() );

	Mat mres;
	mres.push_back(h_hist);
	mres.push_back(s_hist);
	mres.push_back(v_hist);
	mres.push_back(b_hist);
	mres.push_back(g_hist);
	mres.push_back(r_hist);
	mres=mres.t();
	return mres;
}

Mat fftHist(Mat image)
{
	Mat gray;
	//gray=image;
	cvtColor( image, gray, CV_BGR2GRAY );
	Mat padded;                           
    int m = getOptimalDFTSize( gray.rows );
    int n = getOptimalDFTSize( gray.cols );
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);      
    dft(complexI, complexI);    
	split(complexI, planes);            
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    magI += Scalar::all(1); 
    log(magI, magI);
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));  
    Mat q1(magI, Rect(cx, 0, cx, cy)); 
    Mat q2(magI, Rect(0, cy, cx, cy));  
    Mat q3(magI, Rect(cx, cy, cx, cy)); 
    Mat tmp;           
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                  
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, CV_MINMAX);

	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	calcHist( &magI, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
	normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
	hist=hist.t();
	return hist;
}


int classify(Mat image)
{
	CvSVM SVM;
	SVM.load("svm_lp");
	int clres = -8;
	Mat img2cl;
	img2cl=getHist(image);
	clres=SVM.predict(img2cl);
	return clres;
}

int main()
{
	String dir="tests/";
	//String dir="Dataset/";
	String filename = "test01.jpg";
	//String filename = "2014-04-10 14.36.30.jpg";
	Mat img = imread(dir+filename);
	Mat gray,canny;
	cvtColor(img,gray,CV_BGR2GRAY);
	blur( gray, gray, Size(3,3) );
	Canny(gray,canny,100,50);
	Mat bin;
	threshold(gray,bin,120,255,THRESH_BINARY);
	//imshow("Canny",canny);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//findContours(canny,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	findContours(bin,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	for( int i = 0; i< contours.size(); i++ )
	{
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		//approxPolyDP(cv::Mat(contours[i]), contours_poly[i], arcLength(Mat(contours[i]), true)*0.02, true);
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	}

	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
	int indx=0;
	vector<Mat> imagens;
	vector<String>nomes;
	//int auxind=0;
	for( int i = 0; i< contours.size(); i++ )
    {
		Scalar color = Scalar( 255, 255, 255 );
		Scalar color2 = Scalar( 0, 0, 255 );
		Scalar color3 = Scalar( 0, 255, 0 );
		Scalar color4 = Scalar( 255, 0, 0 );
		//drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		if(contours_poly[i].size()>=4&&contours_poly[i].size()<=20&&contourArea(contours[i])>1000)//&&isContourConvex(contours_poly[i])&&contourArea(contours[i])>80)
		{
			drawContours( drawing, contours_poly, i, color, -1, 8, vector<Vec4i>(), 0, Point() );
			//rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color2, 2, 8, 0 ); //plates
			/*Mat temp=Mat::zeros(drawing.size(),CV_8UC3);
			drawContours( temp, contours_poly, i, color, -1, 8, vector<Vec4i>(), 0, Point() );
			img.copyTo(temp,temp);*/
			stringstream s;s <<filename<< " " << indx;
			//cout<<s.str()<<endl;
			/*if(indx==5||indx==8)
			{
				stringstream s2; s2<<auxind;
				imwrite("./Dataset/positivos/"+s2.str()+".jpg",img(boundRect[i]));
				auxind++;
			}*/
			indx++;
			imagens.push_back(img(boundRect[i]));
			nomes.push_back(s.str());
		}
		
		//if(boundRect[i])
		/*else
			rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color3, 2, 8, 0 );*///bounded box of non plates
    }
	namedWindow( "Contours", CV_WINDOW_NORMAL );
	img.copyTo(drawing,drawing);
	Mat aux,aux2;
	cvtColor(drawing,aux,CV_BGR2GRAY);
	threshold(aux,aux2,120,255,THRESH_BINARY);
	blur( aux2, aux2, Size(2,2) );
	//Canny(aux2,aux2,100,50);
	imshow( "Contours", aux2 );
	for(int i =0;i<imagens.size();i++)
	{
		imshow(nomes[i],imagens[i]);
		int cl = classify(imagens[i]);
		if(cl==1)
		{
			cout<<nomes[i]<<": "<<"is plate!"<<endl;
		}
		else
		{
			cout<<nomes[i]<<": "<<"is NOT plate!"<<endl;
		}
	}
	
	imshow("Imagem",img);
	//imshow("Gray",gray);

	Mat ocr=aux2;
	//cvtColor(drawing,ocr,CV_BGR2GRAY);
    tesseract::TessBaseAPI tess;
	tess.Init( NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode( tesseract::PSM_SINGLE_BLOCK );
	tess.SetImage((uchar*)ocr.data,ocr.cols,ocr.rows,1,ocr.cols);
	char* outc = tess.GetUTF8Text();
	string ou = String(outc);
	cout<<ou<<endl;

	waitKey(0);
	return 0;
}