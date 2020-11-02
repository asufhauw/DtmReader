

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


double dot(Vec4f u, Vec4f v)
{
    double sigma = 0;
    for (size_t i = 0; i < 4; i++)
    {
        sigma += u[i] * v[i];
    }
    return sigma;
}
double pi = 3.1415926535897;
bool isPerp(Vec4f line1, Vec4f line2)
{

    double dot1 = dot(line1, line2);
    double dot2 = dot(Vec4f(0,0,1,0), Vec4f(0, 0, 0, 1));
    std::cout << dot1 << endl;
    std::cout << dot2 << endl;
    if (dot1 < 0.01)
        return true;
    else
    {
        double abs_line1 = dot(line1, line1);
        std::cout << "||line1|| == "<< std::sqrt(abs_line1) << std::endl;
        double abs_line2 = dot(line2, line2);
        std::cout <<"||line2|| == "<< std::sqrt(abs_line2) << std::endl;
        double angle = std::acos(dot1 / (abs_line1 * abs_line2));
        std::cout << "Vinkel: " << angle << std::endl;
        if (abs(angle - pi / 2) < 0.01)
        {
            std::cout << abs(angle - pi / 2) << std::endl;
            return true;
        }
         
    }
    return false;


}

#include <opencv2/ximgproc/fast_line_detector.hpp>
void detector(cv::Mat img, cv::Mat oImg)
{
    // Create FLD detector
    // Param               Default value   Description
    // length_threshold    10            - Segments shorter than this will be discarded
    // distance_threshold  1.41421356    - A point placed from a hypothesis line
    //                                     segment farther than this will be
    //                                     regarded as an outlier
    // canny_th1           50            - First threshold for
    //                                     hysteresis procedure in Canny()
    // canny_th2           50            - Second threshold for
    //                                     hysteresis procedure in Canny()
    // canny_aperture_size 3             - Aperturesize for the sobel
    //                                     operator in Canny()
    // do_merge            false         - If true, incremental merging of segments
    //                                     will be perfomred
    int length_threshold = 20;
    float distance_threshold = 1.41421356f;
    double canny_th1 = 50;
    double canny_th2 = 50;
    int canny_aperture_size = 3;
    bool do_merge = false;
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    
        
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
    fld->detect(img, lines_fld);
    double duration_ms = double(getTickCount() - start) * 1000 / freq;
    std::cout << "Elapsed time for FLD " << duration_ms << " ms." << std::endl;
    for (size_t i = 0; i < lines_fld.size()-1; i++)
    {
      //  int x_1 = lines_fld[i];
        std::cout << lines_fld[i] << std::endl;
        std::cout << lines_fld[i+1] << std::endl;
        isPerp(lines_fld[i], lines_fld[i + 1]);
        std::vector<cv::Vec4f> lines;
        lines.push_back(lines_fld[i]);
        lines.push_back(lines_fld[i+1]);

        Mat line_image_fld(img);
        fld->drawSegments(line_image_fld, lines);
        cv::imshow("FLD result", line_image_fld);
        cv::waitKey(0);
    }

   // }
    // Show found lines with FLD
    Mat line_image_fld(img);
    fld->drawSegments(line_image_fld, lines_fld);
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
    


    // The captured image and its grey conversion
    Mat image, grey;

    // Open output window
    namedWindow("ZXing", cv::WINDOW_AUTOSIZE);

    // Stopped flag will be set to -1 from subsequent wayKey() if no key was pressed
    int stopped = -1;

    //set the callback function for any mouse event
    cv::setMouseCallback("window", CallBackFunc, &image);
  
    // Capture image
    image = cv::imread("C:/Users/dajo/source/repos/Project3/x64/Debug/datamatrix2.jpg"); //videoCapture.read(image);
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
