#include <bits/stdc++.h>
#define bug(x) cout << #x << " = " << (x) << '\n'
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <eigen3/Eigen/Dense>


using namespace cv;
using namespace std;
using namespace Eigen;

using tensor = vector<Mat>;
using kernel = tensor;

Rect2i cut_image(const Mat &frame)
{
    Mat _frame;
    frame.copyTo(_frame);
    putText(_frame, "Minimum size", Point(10, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 255, 0));
    rectangle(_frame, Rect(20, 20, 21, 21), Scalar(255, 0, 0));
    Rect2i roi = selectROI("Tracker", _frame, false, false);

    roi.width = max(roi.width, 21), roi.height = max(roi.height, 21);

    if(roi.width % 2 == 0) roi.width++;
    if(roi.height % 2 == 0) roi.height++;

    return roi;
}
tensor get_features(const Mat &frame, const Rect2i &roi)
{
    Mat img = frame(roi), greyImg, dx, dy, d2x, d2y;
    tensor features;
    split(img, features);
    cvtColor(img, greyImg, CV_BGR2GRAY);
    Sobel(greyImg, dx, CV_32FC1, 1, 0, 1);
    Sobel(greyImg, dy, CV_32F, 0, 1, 1);
    Sobel(greyImg, d2x, CV_32F, 2, 0, 1);
    Sobel(greyImg, d2y, CV_32F, 0, 2, 1);

    features.push_back(dx);
    features.push_back(dy);
    features.push_back(d2x);
    features.push_back(d2y);
    return features;
}

Mat get_cov(const tensor &roi_tensor)
{

    int m = roi_tensor[0].rows, n = roi_tensor[0].cols, sz = roi_tensor.size();
    Mat cov = Mat::zeros(sz, sz, CV_64F), mean = Mat::zeros(1, sz, CV_64F), aux = Mat::zeros(1, sz, CV_64F);

    for(int i = 0; i < roi_tensor.size(); ++i)
        mean.at<double>(0, i) = sum(roi_tensor[i])[0] / (m * n);
    bug(mean);
    for(int i = 0;  i < m; ++i)
        for(int j = 0; j < n; ++j)
        {
            for(int k = 0; k < sz; ++k)
                aux.at<double>(0, k) = roi_tensor[k].at<double>(i, j);
            aux = (aux - mean);
            cov += (aux.t()) * aux;
        }
    bug(cov);
    return cov;
}

int main(int argc, char** argv)
{

    VideoCapture cap;
    if(argc < 2)
        cap = VideoCapture(0);
    else
        cap = VideoCapture(argv[1]);
    if(!cap.isOpened())
        return -1;

    Mat frame;
    int x, y;
    Rect2i roi;
    /*if(argc < 2)
        for(int i = 0; i < 20; ++i) cap >> frame;
    cap >> frame;
    */
    tensor feat;
    for(;;)
    {
        cap >> frame;
        if(waitKey(1) == 't')
        {
            roi = selectROI("First", frame, false, false);
            x = roi.x + roi.width / 2 + 1, y = roi.y + roi.height / 2 + 1;
            feat = get_features(frame, roi);
            for(auto i: feat) bug(i);

            for(size_t i = 0; i < feat.size(); ++i)
                imshow("Feature" + to_string(i), feat[i]);
            rectangle(frame, roi, Scalar(0, 255, 0), 0.5);
            circle(frame, Point2i(x, y), 1.5, Scalar(255, 0, 0), 2);
            get_cov(feat);
        }


        imshow("Tracking", frame);
        if(waitKey(30) == 27) break;
    }
    return 0;
}
