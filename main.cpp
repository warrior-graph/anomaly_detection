#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}


int main( int argc, char** argv )
{
    //ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 200;
    bool needToInit = false;
    bool nightMode = false;
    bool alwaysOn = false;
    int time = 0;
    cv::CommandLineParser parser(argc, argv, "{@input||}{help h||}");
    string input = parser.get<string>("@input");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    if( input.empty() )
        cap.open(0);
    else if( input.size() == 1 && isdigit(input[0]) )
        cap.open(input[0] - '0');
    else
        cap.open(input);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }
    cap.open("/home/marques/Workspace/kanade/outputfile.mp4");

    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse, 0 );

    Mat gray, prevGray, image, frame;
    vector<Point2f> points[2];
    //vector<float> dists;
    //double mean, deviation;
    Scalar mean, deviation;
    Mat dists;
    double Th1, Th2, distAux;
    //int fps = cap.get(CV_CAP_PROP_FPS) + .5;
    for(;;)
    {
        cap >> frame;
        if( frame.empty() )
        {
            cap.open("/home/marques/Workspace/kanade/outputfile.mp4");
            needToInit = true;
            continue;
            //break;
        }

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if( needToInit )
        {
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            //line(image, Point(102, 100), Point(255, 300),Scalar(0, 0, 155));
            //for(int i = 0; i < points[0].size(); ++i)
            //{
                //line(image, Point((int) points[0][i].x, (int)points[0][i].y), Point((int) points[1][i].x,(int) points[1][i].y), Scalar(0, 0, 255));
                //cout << points[0][i].x << "  " << points[0][i].y << '\n';
            //}
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                circle( image, points[1][i], 2, Scalar(0,255,0), -1, 8);
                arrowedLine(image, Point((int)points[0][i].x, (int)points[0][i].y),
                                   Point((int)points[1][i].x, (int)points[1][i].y), Scalar(255, 100, 255), 1.2, CV_AA, 0, 0.6);
                dists.push_back(norm(points[0][i] - points[1][i]));

            }
            points[1].resize(k);

            if(points[1].size() < MAX_COUNT / 2)
            {
                needToInit = true;
                continue;
            }

            meanStdDev(dists, mean, deviation);
            Th1 = mean[0] + deviation[0];
            Th2 = mean[0] - deviation[1];
            //waitKey(50);
            for(i = 0; i < dists.size[0]; ++i)
            {
                distAux = dists.at<double>(i);
                //cout << distAux << '\n';
                //if(distAux < 0. or fabs(distAux) < 10) continue;
                if ((distAux > Th1))
                {
                     circle( image, points[1][i], 2, Scalar(0,0,255), 3, 8);
                     //arrowedLine(image, Point((int)points[0][i].x, (int)points[0][i].y),
                                        //Point((int)points[1][i].x, (int)points[1][i].y), Scalar(255, 0, 0), 1.2, CV_AA, 0, 0.6);
                }
            }
           // cout << dists.size[0] << '\n';

            //cout << dists.at<double>(1) << ' ' << deviation[0] << ' ' << mean[0] << endl;
            //cout << "dist size = " << dists.size() << endl;

            dists.~Mat();
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }
        putText(image, "Threshold 1 = " + to_string(Th1), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        line(image, Point(10, 40), Point(10 + Th1 , 40), Scalar(0, 255, 0), 2);
        putText(image, "Threshold 2 = " + to_string(Th2), Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        line(image, Point(10, 90), Point(10 + Th2 , 90), Scalar(0, 255, 255), 2);

        needToInit = false;
        imshow("LK Demo", image);
        time++;

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        if(alwaysOn and time % 2 == 0)
            time = 0, c = 'r';
        switch( c )
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        case 'a':
            alwaysOn = !alwaysOn;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
        //waitKey(100);
    }

    return 0;
}
