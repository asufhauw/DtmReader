


#include <string>
#include <exception>
#include <stdlib.h>
#include <zxing/common/Counted.h>
#include <zxing/Binarizer.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/ReaderException.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/Exception.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
//#include <zxing/qrcode/QRCodeReader.h>
//#include <zxing/MultiFormatReader.h>
#include <zxing/MatSource.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <zxing/datamatrix/DataMatrixReader.h>
#include <zxing/datamatrix/detector/Detector.h>
using namespace std;
using namespace zxing;
using namespace cv;

double pi = 3.1415926535897;
#if CV_MAJOR_VERSION >= 4
#ifndef CV_CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_WIDTH CAP_PROP_FRAME_WIDTH
#endif
#ifndef CV_CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FRAME_HEIGHT CAP_PROP_FRAME_HEIGHT
#endif
#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY COLOR_BGR2GRAY
#endif
#endif

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if (event == cv::EVENT_MBUTTONDOWN)
    {
        cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

    }
}

cv::Point startP1 = cv::Point(0, 0);
cv::Point endP2 = cv::Point(0, 0);
void getPoints(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        startP1 = cv::Point(x, y);
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {

        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl; //  << img->at<ushort>(y, x)
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        cout << "Left button of the mouse is Up - position (" << x << ", " << y << ")" << endl;
        endP2 = cv::Point(x, y);
        cv::Mat* img = static_cast<cv::Mat*>(userdata);
        cv::Rect rect1(startP1, endP2);
        *img = cv::Mat(*img, rect1);
    }
}

void printUsage(char** argv) {

    // Print usage
    cout << "Usage: " << argv[0] << " [-d <DEVICE>] [-w <CAPTUREWIDTH>] [-h <CAPTUREHEIGHT>]" << endl
        << "Read QR code from given video device." << endl
        << endl;

}

Point toCvPoint(Ref<ResultPoint> resultPoint) {
    return Point(resultPoint->getX(), resultPoint->getY());
}

void decoder(cv::Mat image)
{


    // Create luminance  source
    try
    {
        zxing::Ref<zxing::LuminanceSource> source = MatSource::create(image);
        zxing::Ref<zxing::Reader> reader;
        reader.reset(new zxing::datamatrix::DataMatrixReader);
        zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));//HybridBinarizer GlobalHistogramBinarizer
        zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));
        zxing::DecodeHintType hint = zxing::DecodeHints::DATA_MATRIX_HINT | zxing::DecodeHints::TRYHARDER_HINT; // |  zxing::DecodeHints::TRYHARDER_HINT
        //cv::imshow("window", bitmap->getBlackMatrix());
        zxing::Ref<zxing::Result> result(reader->decode(bitmap, zxing::DecodeHints(hint)));

        // Get result point count
        int resultPointCount = result->getResultPoints()->size();

        for (int j = 0; j < resultPointCount; j++) {

            // Draw circle
            cv::circle(image, toCvPoint(result->getResultPoints()[j]), 0, cv::Scalar(110, 220, 0), 2);

        }
       
        // Draw boundary on image
        if (resultPointCount > 1) {
            for (int j = 0; j < resultPointCount; j++) {

                // Get start result point
                zxing::Ref<zxing::ResultPoint> previousResultPoint = (j > 0) ? result->getResultPoints()[j - 1] : result->getResultPoints()[resultPointCount - 1];

                // Draw line
                cv::line(image, toCvPoint(previousResultPoint), toCvPoint(result->getResultPoints()[j]), cv::Scalar(110, 220, 0), 2, 8);

                // Update previous point
                previousResultPoint = result->getResultPoints()[j];

            }

        }
        if (resultPointCount > 0) {
            // Draw text
            cv::putText(image, result->getText()->getText(), toCvPoint(result->getResultPoints()[0]), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(110, 220, 0));
            cout << result->getText()->getText() << endl;
        }

        cv::imshow("window", image);
        cv::waitKey(0);
    }
    catch (const zxing::ReaderException& e) {
        cerr << e.what() << " (ignoring)" << endl;

    }
    catch (const zxing::IllegalArgumentException& e) {
        cerr << e.what()  << " (ignoring)" << endl;
    }
    catch (const zxing::Exception& e) {
        cerr << e.what() << " (ignoring)" << endl;
    }
    catch (const std::exception& e) {
        cerr << e.what() << " (ignoring)" << endl;
    }


}


void fixImage(cv::Mat img,cv::Mat oImg)
{
    // Find all contours in the image
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    cv::Mat imgGray, imgThres;

    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::threshold(imgGray, imgThres, 100, 255, cv::THRESH_BINARY);
    // Find all contours in the image
    findContours(imgThres, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f rect_points[4];
    cv::Mat boxPoints2f, boxPointsCov;
    cv::Rect rect;
    for (size_t i = 0; i < contours.size(); i++) {
        // Vertical rectangle
        if (contours[i].size() < 200) continue;

        rect = boundingRect(contours[i]);

        cv::Mat boxed = cv::Mat(imgGray, rect);

        cv::Point centerpoint = cv::Point(boxed.cols / 2, boxed.rows / 2);

        double angle = minAreaRect(contours[i]).angle;
        double scale = 1;
        cv::Mat rot_mat = getRotationMatrix2D(centerpoint, angle, scale);
        cv::Mat warp_rotate_dst;
        warpAffine(boxed, boxed, rot_mat, boxed.size());
        //cv::waitKey(0);
        cv::imshow("test", boxed);

        cv::waitKey(0);
        oImg = boxed.clone();
        return;
    }

}


Vec2f norm(Vec4f u)
{
    return Vec2f(u[2] - u[0], u[3] - u[1]);
}

double Vec4fLength(Vec4f u);
Vec4f normVec4f(Vec4f u)
{
    double l = Vec4fLength(u);
    std::cout << l << std::endl;
    return Vec4f(u[0] - l, u[1] - l, u[2] - l, u[3] - l);

}

double dot(Vec4f u, Vec4f v)
{
    Vec4f norm_u = normVec4f(u);
    Vec4f norm_v = normVec4f(v);

    double sigma = 0;
    for (size_t i = 0; i < 4; i++)
    {
        sigma += norm_u[i] * norm_v[i];
    }
    return sigma;
}

double dot(Vec2f u, Vec2f v)
{
    double sigma = 0;
    for (size_t i = 0; i < 2; i++)
    {
        sigma += u[i] * v[i];
    }
    return sigma;
}

double Vec4fLength(Vec4f u)
{
    Vec2f dest(u[0] - u[2], u[1] - u[3]);
    return sqrt(dot(dest, dest));
}
double Vec2fLength(Vec2f u)
{
    return sqrt(dot(u,u));
}
double getAngle(Vec2f u)
{
    return atan(u[1] / u[0]);
}
double getAngle(Vec2f line1, Vec2f line2)
{

    double dot1 = dot(line1, line2);
    double abs_line1 = dot(line1, line1);
    double abs_line2 = dot(line2, line2);
    double angle = std::acos(dot(line1, line2) / sqrt(dot(line1, line1) * dot(line2, line2)));
    std::cout << "angle: " << angle << "radians\t" << angle * 360 / (2 * pi) << "degrees" << std::endl;
    return angle;
}
double getAngle(Vec4f line1, Vec4f line2)
{

 //   double dot1 = dot(line1, line2);
 //   double abs_line1 = dot(line1, line1);
 //   double abs_line2 = dot(line2, line2);
 //   double angle = std::acos(dot(line1, line2) / sqrt(dot(line1, line1) * dot(line2, line2)));
 //   std::cout << "angle: " << angle << "radians\t" << angle * 360 / (2*pi) << "degrees" << std::endl;
    return getAngle(norm(line1),norm(line2));
}

bool isPerp(Vec4f u, Vec4f v)
{
    double udotv = dot(norm(u), norm(v));
    std::cout << udotv << endl;
    if (udotv < 0.01)
        return true;
    else
    {
        if (abs(getAngle(norm(u), norm(v)) - pi / 2) < 0.01)
            return true;
    }
    return false;
}

Vec4f diffVec4f(Vec4f u, Vec4f v)
{
    return Vec4f(u[0]-v[0], u[1] - v[1], u[2] - v[2], u[3] - v[3]);
}

Vec2f pointTheSame(Vec4f u, Vec4f v)
{
    Vec2f u_p1(u[0], u[1]);
    Vec2f u_p2(u[2], u[3]);
    Vec2f v_p1(v[0], v[1]);
    Vec2f v_p2(v[2], v[3]);
    if (dot(u_p1 - v_p1, u_p1 - v_p1) < 10)
        return u_p1;
    else if (dot(u_p2 - v_p1, u_p2 - v_p1) < 10)
        return u_p2;
    else if (dot(u_p2 - v_p2, u_p2 - v_p2) < 10)
        return u_p2;
    else if (dot(u_p1 - v_p2, u_p1 - v_p2) < 10)
        return u_p1;
    return Vec2f(-1,-1);
}
bool linesClose(Vec4f u, Vec4f v)
{
    Vec2f u_p1(u[0], u[1]);
    Vec2f u_p2(u[2], u[3]);
    Vec2f v_p1(v[0], v[1]);
    Vec2f v_p2(v[2], v[3]);
    double l1 = (dot(u_p1 - v_p1, u_p1 - v_p1));
    double l2 = (dot(u_p2 - v_p1, u_p2 - v_p1));
    double l3 = (dot(u_p2 - v_p2, u_p2 - v_p2));
    double l4 = (dot(u_p1 - v_p2, u_p1 - v_p2));
    if (l1<10)
        return true;
    else if (l2 < 10)
        return true;
    else if (l3 < 10)
        return true;
    else if (l4 < 10)
        return true;
    return false;
    /*
    if (abs(u[0] - v[0]) < 3 && abs(u[1] - v[1]) < 3)
        return true;
    else if (abs(u[2] - v[0]) < 3 && abs(u[3] - v[1]) < 3)
        return true;
    else if (abs(u[2] - v[2]) < 3 && abs(u[3] - v[3]) < 3)
        return true;
    return false;
    */
    
}

Vec2f mulPointMat(cv::Mat m, Vec2f p)
{
    return Vec2f(m.at<double>(0, 0) * p[0]+ m.at<double>(0, 1)*p[1], m.at<double>(1, 0) * p[0] + m.at<double>(1, 1) * p[1]);
}

#include <opencv2/ximgproc/fast_line_detector.hpp>

void detector(cv::Mat img, cv::Mat oImg)
{
    int length_threshold = 30;
    float distance_threshold = 1.41421356f;
    double canny_th1 = 50;
    double canny_th2 = 50;
    int canny_aperture_size = 3;
    bool do_merge = false;
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    Mat thresholdImg;
    cv::threshold(img, thresholdImg, 100, 255, cv::THRESH_BINARY);
   // cv::imshow("FLD result", thresholdImg);
   // cv::imshow("FLD result1", img);
   // cv::waitKey(0);
    Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(length_threshold,
        distance_threshold, canny_th1, canny_th2, canny_aperture_size,
        do_merge);;

    vector<Vec4f> lines_fld;
    // Because of some CPU's power strategy, it seems that the first running of
    // an algorithm takes much longer. So here we run the algorithm 10 times
    // to see the algorithm's processing time with sufficiently warmed-up
    // CPU performance.
   // for (int run_count = 0; run_count < 10; run_count++) {
    double freq = getTickFrequency();
    lines_fld.clear();
    int64 start = getTickCount();
    // Detect the lines with FLD
    fld->detect(thresholdImg, lines_fld);
    double duration_ms = double(getTickCount() - start) * 1000 / freq;
    std::cout << "Elapsed time for FLD " << duration_ms << " ms." << std::endl;
    std::vector<cv::Vec2f> lines;
    for (size_t i = 0; i < lines_fld.size()-1; i++)
    {
        for (size_t j = i+1; j < lines_fld.size(); j++)
        {
            Vec4f u_hat = lines_fld[i];
            Vec4f v_hat = lines_fld[j];

            if (linesClose(u_hat,v_hat) && (abs(getAngle(u_hat, v_hat)- pi/2)<10)) //&& abs(Vec4fLength(u_hat) - Vec4fLength(v_hat))<20
            {
              
                lines.push_back(Vec2f(u_hat[0], u_hat[1]));
                lines.push_back(Vec2f(u_hat[2], u_hat[3]));
                lines.push_back(Vec2f(v_hat[0], v_hat[1]));
                lines.push_back(Vec2f(v_hat[2], v_hat[3]));
                Vec2f norm_u = norm(u_hat);
                double rotAngle = getAngle(norm_u);
                // make roi
                std::vector<Vec2f> points;
                points.push_back(Vec2f(u_hat[0], u_hat[1]));
                points.push_back(Vec2f(u_hat[2], u_hat[3]));
                points.push_back(Vec2f(v_hat[0], v_hat[1]));
                points.push_back(Vec2f(v_hat[2], v_hat[3]));

                Vec4f roi(norm(u_hat)[0], norm(u_hat)[1], norm(v_hat)[0], norm(v_hat)[1]);
                Mat rotMat = getRotationMatrix2D(Point2f(u_hat[0], u_hat[1]), rotAngle*180/pi, 1);
                
                for (size_t i = 0; i < points.size(); i++)
                {
                    points[i] = mulPointMat(rotMat,points[i]);
                }

                for (size_t i = 0; i < lines.size(); i++)
                {
                    ;// lines[i] = mulPointMat(rotMat, lines[i]);
                }
                std::vector<Vec4f> l;
                for (size_t i = 0; i < lines.size()-2; i+=2)
                {
                    l.push_back(Vec4f(points[i][0], points[i][1], points[i+1][0], points[i+1][1]));
                }
                warpAffine(img, thresholdImg, rotMat,img.size());
                
                // decode the shit....
                Mat line_image_fld(thresholdImg);
                fld->drawSegments(line_image_fld, l,false); // lines
                cv::imshow("FLD result1", line_image_fld);
                cv::waitKey(0);
            }
            
        }
      //  int x_1 = lines_fld[i];
       
    }

   // }
    // Show found lines with FLD
    Mat line_image_fld(img);
    fld->drawSegments(line_image_fld, lines);
    cv::imshow("FLD result", line_image_fld);
    cv::waitKey(0);

    oImg = img.clone();
}

int main(int argc, char** argv) {

    int deviceId = 0;
    int captureWidth = 640;
    int captureHeight = 480;
    bool multi = false;

    // Open video captire
   // VideoCapture videoCapture(deviceId);
    Vec4f u(0, 0, 2, 0);
    Vec4f v(0, 0, 0, -2);
    double angle = getAngle(norm(u), norm(v));
    std::cout << "norm u: " << norm(u) << "\tv norm: " << norm(v) << std::endl;

    std::cout << "angle: " << angle << "\tGrader: " << angle * 180 / pi << std::endl;

    // The captured image and its grey conversion
    Mat image, grey;

    // Open output window
    namedWindow("ZXing", cv::WINDOW_AUTOSIZE);

    // Stopped flag will be set to -1 from subsequent wayKey() if no key was pressed
    int stopped = -1;

    //set the callback function for any mouse event
    cv::setMouseCallback("window", CallBackFunc, &image);
  
    // Capture image
    image = cv::imread("C:/Users/johan/Projects/DtmReader/x64/Debug/datamatrix2.jpg"); //datamatrix2.jpg datamatrix.png

    if (image.empty())
        return -1;
   // Show captured image

     
    cv::Mat img;
    // fixImage(image, img);
    detector(image, img);
    // if(img != NULL)
    // decoder(img);
           

    // Release video capture
    //videoCapture.release();

    return 0;

}
