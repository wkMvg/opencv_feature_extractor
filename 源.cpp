#include<iostream>
#include<vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <features2d.hpp>
#include <xfeatures2d.hpp>
#include <flann.hpp>


//#include "openMVG/matching/indMatch.hpp"
//#include "openMVG/matching/indMatch_utils.hpp"
#include <time.h>

//using namespace openMVG;
//using namespace openMVG::matching;

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//#define brisk
//#define surf
//#define freak
//#define harris
//#define sift
#define orb
int main()
{
	int a;
	a = 1 << 18;
	Mat img1 = imread("F:\\data\\111\\DSC00003.JPG");
	Mat img2 = imread("F:\\data\\111\\DSC00266.JPG");

#ifdef  orb
	Ptr<ORB> orbFeat = ORB::create(40000, 1.2f, 2, 31, 0, 2,ORB::FAST_SCORE);
	//Ptr<ORB> orbFeat = ORB::create(20000, 1.2f, 6, 31, 0, 2, ORB::HARRIS_SCORE);

	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;

	Mat descrip1, descrip2;

	Mat img;
	cvtColor(img1, img, COLOR_RGB2GRAY);


	clock_t start_detec = clock();
	orbFeat->detectAndCompute(img1, Mat(), keypoint1, descrip1);
	clock_t end_detec = clock();
	cout << "detect cost time is: " << end_detec - start_detec << endl;
	orbFeat->detectAndCompute(img2, Mat(), keypoint2, descrip2);


	clock_t start = time(NULL);
	// 最近邻点和次近邻点匹配
	BFMatcher matcher2(NORM_HAMMING);
	Ptr<FlannBasedMatcher> matcher3 = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 1));
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;

	
	//matcher2.knnMatch(descrip1, descrip2, matches_nn, 2);
	clock_t t2 = clock();
	matcher3->knnMatch(descrip1, descrip2, matches_nn, 2);
	clock_t t1 = clock();
	std::cout << "knn Match cost " << t1 - t2 << endl;
	vector<vector<DMatch>> matches_ann;
#pragma omp parallel for
	for (int i = 0; i < matches_nn.size() && matches_nn[i].size() == 2; i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif //  ORB

#ifdef brisk
	Ptr<BRISK> orbFeat = BRISK::create(40, 3, 1.f);

	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;

	Mat descrip1, descrip2;

	clock_t start_detec = time(NULL);
	orbFeat->detectAndCompute(img1, Mat(), keypoint1, descrip1);
	clock_t end_detec = time(NULL);
	cout << "detect cost time is: " << end_detec - start_detec << endl;
	orbFeat->detectAndCompute(img2, Mat(), keypoint2, descrip2);


	clock_t start = time(NULL);
	// 最近邻点和次近邻点匹配
	Ptr<DescriptorMatcher> matrcher2 = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;
	matrcher2->knnMatch(descrip1, descrip2, matches_nn, 2);
//#pragma omp parallel for
	for (int i = 0; i < matches_nn.size(); i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif	
#ifdef surf
	Ptr<SURF> orbFeat = SURF::create(400);

	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;

	Mat descrip1, descrip2;
	clock_t start_detec = time(NULL);
	orbFeat->detectAndCompute(img1, Mat(), keypoint1, descrip1);
	clock_t end_detec = time(NULL);
	std::cout << "detect cost time is: " << end_detec - start_detec << endl;
	orbFeat->detectAndCompute(img2, Mat(), keypoint2, descrip2);
	std:cout << "descriptors type is" << descrip2.type() << endl;


	clock_t start = time(NULL);
	// 最近邻点和次近邻点匹配
	Ptr<DescriptorMatcher> matrcher2 = DescriptorMatcher::create("FlannBased");
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;
	matrcher2->knnMatch(descrip1, descrip2, matches_nn, 2);
	//#pragma omp parallel for
	for (int i = 0; i < matches_nn.size(); i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif	

#ifdef freak
	Ptr<FREAK> orbFeat = FREAK::create();

	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;

	Mat descrip1, descrip2;
	clock_t start_detec = time(NULL);
	orbFeat->detectAndCompute(img1, Mat(), keypoint1, descrip1);
	clock_t end_detec = time(NULL);
	std::cout << "detect cost time is: " << end_detec - start_detec << endl;
	orbFeat->detectAndCompute(img2, Mat(), keypoint2, descrip2);
    std:cout << "descriptors type is" << descrip2.type() << endl;


	clock_t start = time(NULL);
	// 最近邻点和次近邻点匹配
	Ptr<DescriptorMatcher> matrcher2 = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;
	matrcher2->knnMatch(descrip1, descrip2, matches_nn, 2);
	//#pragma omp parallel for
	for (int i = 0; i < matches_nn.size(); i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif

#ifdef harris
	cv::cvtColor(img1, img1, COLOR_RGB2GRAY);
	cv::cvtColor(img2, img2, COLOR_RGB2GRAY);
	Mat harrisCorner1, harrisCorner2;

	clock_t start = time(NULL);
	cv::cornerHarris(img1, harrisCorner1, 2, 3, 0.04);
	cv::cornerHarris(img2, harrisCorner2, 2, 3, 0.04);

	Mat dst_norm1, dst_norm2;
	normalize(harrisCorner1, dst_norm1, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(harrisCorner1, dst_norm2, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	vector<KeyPoint> keypoints1, keypoints2;
	for (int i = 0; i < dst_norm1.rows; i++)
	{
		for (int j = 0; j < dst_norm1.cols; j++)
		{
			if ((int)dst_norm1.at<float>(i, j) > 70)
			{
				KeyPoint keypoint1;
				keypoint1.pt.x = j;
				keypoint1.pt.y = i;
				keypoints1.push_back(keypoint1);
			}
			if ((int)dst_norm2.at<float>(i, j) > 70)
			{
				KeyPoint keypoint2;
				keypoint2.pt.x = j;
				keypoint2.pt.y = i;
				keypoints2.push_back(keypoint2);
			}
		}
	}
	Mat descrip1, descrip2;
	Ptr<BRISK> surfDet = BRISK::create();
	surfDet->compute(img1, keypoints1, descrip1);
	surfDet->compute(img2, keypoints2, descrip2);
	clock_t end = time(NULL);
	cout << "detect cost time is: " << end - start << endl;
//	// 最近邻点和次近邻点匹配
	Ptr<DescriptorMatcher> matrcher2 = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;
	matrcher2->knnMatch(descrip1, descrip2, matches_nn, 2);
	//#pragma omp parallel for
	for (int i = 0; i < matches_nn.size(); i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif

#ifdef sift
	Ptr<SIFT> orbFeat = SIFT::create(20000);

	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;

	Mat descrip1, descrip2;
	clock_t start_detec = time(NULL);
	orbFeat->detectAndCompute(img1, Mat(), keypoint1, descrip1);
	clock_t end_detec = time(NULL);
	std::cout << "detect cost time is: " << end_detec - start_detec << endl;
	orbFeat->detectAndCompute(img2, Mat(), keypoint2, descrip2);
std:cout << "descriptors type is" << descrip2.type() << endl;


	clock_t start = time(NULL);
	// 最近邻点和次近邻点匹配
	Ptr<DescriptorMatcher> matrcher2 = DescriptorMatcher::create("FlannBased");
	vector<vector<DMatch>> matches_nn;
	vector<DMatch> goodMatches_nn;
	const float ratio = 0.7;
	matrcher2->knnMatch(descrip1, descrip2, matches_nn, 2);
	//#pragma omp parallel for
	for (int i = 0; i < matches_nn.size(); i++)
	{
		DMatch& bestMatch = matches_nn[i][0];
		DMatch& betterMatch = matches_nn[i][1];
		if (bestMatch.distance < ratio * betterMatch.distance)
		{
			goodMatches_nn.push_back(bestMatch);
		}
	}
#endif
	Mat img_matches_nn;
	drawMatches(img1, keypoint1, img2, keypoint2,
		goodMatches_nn, img_matches_nn, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("match_orb_fastScore_0.8.jpg", img_matches_nn);
	//clock_t end = time(NULL);
	cout << goodMatches_nn.size() << endl;

	while (1);
}