#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

namespace ML{
vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

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
	normalize(Hogfeat,Hogfeat);
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


void write_dataset()
{
	ofstream log;
	log.open("Dataset/lista.csv",std::ios_base::app);
	for(int i=0;i<26;i++)
	{
		stringstream s;
		s<<i;
		log<<"Dataset/negativos/"+s.str()+".jpg;no"<<endl;
	}
	for(int i=0;i<14;i++)
	{
		stringstream s;
		s<<i;
		log<<"Dataset/positivos/"+s.str()+".jpg;yes"<<endl;
	}
	log.close();
}

int trainModel()
{
	ifstream imgdet;
	ofstream log;
	imgdet.open("Dataset/lista.csv");
	string l;
	stringstream filename;
	vector<string> res;
	getline(imgdet,l,'\n');
	Mat features,image,labels;
	int cnt=1;
	cout<<"Processing Images...\n";
	//cnt<251&&
	while(getline(imgdet,l,'\n'))
	{
		cnt++;
		res=split(l,';');
		filename.str(string());
		filename<<res[0];
		if(res[1]=="yes")
		{
			labels.push_back(1);
		}
		else if(res[1]=="no")
			{
				labels.push_back(0);
			}
		else
		{
			log.open("errorlog.txt",ios::app);
			log<<"Unkown label type error!\n";
			log.close();
			continue;
		}
		image=imread(filename.str(),CV_LOAD_IMAGE_COLOR);
		if(!image.data)
		{
			log.open("errorlog.txt",ios::app);
			log<<"Error opening file: "<<filename.str()<<".\n";
			log.close();
			continue;
		}
		features.push_back(getHist(image));

	}
	if(!features.data)
	{
		log.open("errorlog.txt",ios::app);
		log<<"NO FEATURES DETECTED!\n";
		cout<<"ERROR! PLEASE SEE ERRORLOG.TXT FOR DETAILS!\n";
		imgdet.close();
		log.close();
		return -1;
	}
	cout<<"Building Model...\n";
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;
    SVM.train(features, labels, Mat(), Mat(), params);
	cout<<"Saving Model...\n";
	SVM.save("svm_lp");
	cout<<"Done!\n";
	imgdet.close();
	log.close();
	return 0;
}

}

int mainOFF(){
	ML::trainModel();
	//Mat img=imread("Dataset/negativos/1.jpg");
	//ML::hog(img);
	getchar();
	return 0;
}