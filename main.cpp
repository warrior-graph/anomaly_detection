#include <bits/stdc++.h>
#define bug(x) cout << #x << " = " << (x) << '\n'
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <new_eigen.hpp>

using namespace cv;
using namespace std;
using namespace new_eigen;

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
    Mat img = frame(roi), greyImg, d[4], aux(img.rows, img.cols, CV_64F), taux[3];
    tensor features;
    split(img, taux);

    cvtColor(img, greyImg, CV_BGR2GRAY);
    Sobel(greyImg, d[0], CV_64F, 1, 0, 1);
    Sobel(greyImg, d[1], CV_64F, 0, 1, 1);
    Sobel(greyImg, d[2], CV_64F, 2, 0, 1);
    Sobel(greyImg, d[3], CV_64F, 0, 2, 1);
    for(int i = 0; i < 3; ++i)
        taux[i].convertTo(aux, CV_64F), features.push_back(aux), aux.release();
    for(int i = 0; i < 4; ++i)
        d[i].convertTo(aux, CV_64F), features.push_back(aux), aux.release();
    return features;
}


Mat get_cov(const tensor &roi_tensor)
{

    int m = roi_tensor[0].rows, n = roi_tensor[0].cols, sz = roi_tensor.size();
    Mat cov = Mat::zeros(sz, sz, CV_64F),
        mean = Mat::zeros(1, sz, CV_64F),
        aux = Mat::zeros(1, sz, CV_64F),
        aux2 = Mat::zeros(1, sz, CV_64F);

    for(uint i = 0; i < roi_tensor.size(); ++i)
        mean.at<double>(0, i) = sum(roi_tensor[i])[0] / (m * n);
    for(int i = 0;  i < m; ++i)
        for(int j = 0; j < n; ++j)
        {
            for(int k = 0; k < sz; ++k)
                aux.at<double>(0, k) = roi_tensor[k].at<double>(i, j);
            //bug(aux);
            mulTransposed(aux, aux2, true, mean, CV_64F);
            cov += aux2 / (m * n);
        }
    return cov;
}

int main(int argc, char** argv)
{

    VideoCapture cap;
    if(argc < 2)
        cap = VideoCapture(0);
    else
        cap = VideoCapture(argv[1]);
    cap = VideoCapture ("/home/marques/Downloads/vot2014/"+ string(argv[1]) +"/00000%3d.jpg");
    if(!cap.isOpened())
        return -1;

    Mat frame;
    int w_frame = 240, h_frame = 120;
    Rect2i roi, r_track;
    /*if(argc < 2)
        for(int i = 0; i < 20; ++i) cap >> frame;
    cap >> frame;
    */
    tensor feat, track_feat, search_feat;
    vector<Mat> covs, frame_covs;
    Mat model, covalks, modelupdated = Mat::zeros(7, 7, CV_64F);

    char c;
    bool init = false, clean = false;
    for(;;)
    {
        cap >> frame;
        resize(frame, frame, Size(w_frame, h_frame));
        if(init)
        {
            roi = selectROI("First", frame, false, false);

            feat = get_features(frame, Rect2i(0, 0, frame.cols, frame.rows));
            track_feat = get_features(frame, roi);
            //rectangle(frame, roi, Scalar(0, 255, 0), 1);
            //circle(frame, Point2i(x, y), 1.5, Scalar(255, 0, 0), 2);
            covs.push_back(get_cov(feat));
            covs.push_back(get_cov(track_feat));
            init = false;
            continue;
        }
        if(clean) covs.clear(), feat.clear(), clean = false, cout << "tudo limpo\n";

        if(!feat.empty())
        {
            // w_frame - roi.width
            // h_frame - roi.heightteclado lindinh101010
            double dist = 1 << 30, aux;
            for(uint i = 0; i < w_frame - roi.width; i += 15)
                for(uint j = 0; j < h_frame - roi.height; j += 15)
                {
                    search_feat = get_features(frame, Rect2i(i, j, roi.width, roi.height));
                    aux = new_eigen::diss(covalks = get_cov(search_feat), covs[1]);
                    if(aux < dist)
                        dist = aux, r_track = Rect2i(i, j, roi.width, roi.height), model = covalks ;
                    //cout << i << ' ' << j << '\n';
                }
            if(dist == (1 << 30)) {bug("mmorri");break;};
            rectangle(frame, r_track, Scalar(0, 0, 200), 2);
            frame_covs.push_back(model);
            if(frame_covs.size() == 10)
            {
                for(const auto &m: frame_covs)
                    modelupdated += m;
                modelupdated = modelupdated * 0.1;
                //bug(modelupdated);
                covs[1] = modelupdated;
                modelupdated = Mat::zeros(7, 7, CV_64F);
                frame_covs.clear();
            }

        }

        imshow("Tracking", frame);
        //if(waitKey(30) == 27) break;
        c = (char) waitKey(10);
        if(c == 27) break;

        switch (c)
        {
        case 't':
            init = true;
            break;
         case 'c':
            clean = true;
        default:
            break;
        }
    }
    return 0;
}
