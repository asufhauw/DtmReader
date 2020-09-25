/*
 *  Copyright 2010-2011 Alessandro Francescon
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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


int state = 0;
void decoder(cv::Mat image)
{


    // Create luminance  source
    try
    {
        zxing::Ref<zxing::LuminanceSource> source = MatSource::create(image);
        state = 1;
        zxing::Ref<zxing::Reader> reader;
        state = 2;
        reader.reset(new zxing::datamatrix::DataMatrixReader);
        state = 3;
        zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));//HybridBinarizer GlobalHistogramBinarizer
        state = 4;
        zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));
        state = 5;
        zxing::DecodeHintType hint = zxing::DecodeHints::DATA_MATRIX_HINT | zxing::DecodeHints::TRYHARDER_HINT; // |  zxing::DecodeHints::TRYHARDER_HINT
        //cv::imshow("window", bitmap->getBlackMatrix());
        zxing::Ref<zxing::Result> result(reader->decode(bitmap, zxing::DecodeHints(hint)));
        state = 6;

        // Get result point count
        int resultPointCount = result->getResultPoints()->size();
        state = 7;

        for (int j = 0; j < resultPointCount; j++) {

            // Draw circle
            cv::circle(image, toCvPoint(result->getResultPoints()[j]), 0, cv::Scalar(110, 220, 0), 2);

        }
        state = 8;

        // Draw boundary on image
        if (resultPointCount > 1) {
            state = 9;
            for (int j = 0; j < resultPointCount; j++) {

                // Get start result point
                zxing::Ref<zxing::ResultPoint> previousResultPoint = (j > 0) ? result->getResultPoints()[j - 1] : result->getResultPoints()[resultPointCount - 1];

                // Draw line
                cv::line(image, toCvPoint(previousResultPoint), toCvPoint(result->getResultPoints()[j]), cv::Scalar(110, 220, 0), 2, 8);

                // Update previous point
                previousResultPoint = result->getResultPoints()[j];

            }

        }
        state = 10;
        if (resultPointCount > 0) {
            state = 11;
            // Draw text
            cv::putText(image, result->getText()->getText(), toCvPoint(result->getResultPoints()[0]), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(110, 220, 0));
            cout << result->getText()->getText() << endl;
        }

        cv::imshow("window", image);
        cv::waitKey(0);
    }
    catch (const zxing::ReaderException& e) {
        cerr << e.what() << state << " (ignoring)" << endl;

    }
    catch (const zxing::IllegalArgumentException& e) {
        cerr << e.what() << state << " (ignoring)" << endl;
    }
    catch (const zxing::Exception& e) {
        cerr << e.what() << state << " (ignoring)" << endl;
    }
    catch (const std::exception& e) {
        cerr << e.what() << state << " (ignoring)" << endl;
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


int main(int argc, char** argv) {

    int deviceId = 0;
    int captureWidth = 640;
    int captureHeight = 480;
    bool multi = false;

    for (int j = 0; j < argc; j++) {

        // Get arg
        string arg = argv[j];
         
        if (arg.compare("-d") == 0) 
        {
            // Set device id
            if ((j + 1) < argc) { 
                deviceId = atoi(argv[++j]);
            }
            else {
                // Log
                cerr << "Missing device id after -d" << endl;
                printUsage(argv);
                return 1;
            }

        }
        else if (arg.compare("-w") == 0)
        { // Set capture width

            if ((j + 1) < argc) {
                
                captureWidth = atoi(argv[++j]);
            }
            else {
                // Log
                cerr << "Missing width after -w" << endl;
                printUsage(argv);
                return 1;
            }

        }
        else if (arg.compare("-h") == 0) 
        {
            // Set capture height
            if ((j + 1) < argc) {
                captureHeight = atoi(argv[++j]);
            }
            else {
                // Log
                cerr << "Missing height after -h" << endl;
                printUsage(argv);
                return 1;
            }

        }
        else if (arg.compare("-m") == 0) {

            // Set multi to true
            multi = true;

        }

    }

    // Log
    cout << "Capturing from device " << deviceId << "..." << endl;

    // Open video captire
    VideoCapture videoCapture(deviceId);

    if (!videoCapture.isOpened()) {

        // Log
        cerr << "Open video capture failed on device id: " << deviceId << endl;
        return -1;

    }

    if (!videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, captureWidth)) {

        // Log
        cerr << "Failed to set frame width: " << captureWidth << " (ignoring)" << endl;

    }

    if (!videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, captureHeight)) {

        // Log
        cerr << "Failed to set frame height: " << captureHeight << " (ignoring)" << endl;

    }

    // The captured image and its grey conversion
    Mat image, grey;

    // Open output window
    namedWindow("ZXing", cv::WINDOW_AUTOSIZE);

    // Stopped flag will be set to -1 from subsequent wayKey() if no key was pressed
    int stopped = -1;

    //set the callback function for any mouse event
    cv::setMouseCallback("window", CallBackFunc, &image);
    while (stopped == -1) {

        // Capture image
        bool result = videoCapture.read(image);

        // Show captured image

        if (result) {

            cv::Mat img;
            fixImage(image, img);
           // if(img != NULL)
           //     decoder(img);
           
        }
        else {

            // Log
            cerr << "video capture failed" << endl;

        }

    }

    // Release video capture
    videoCapture.release();

    return 0;

}
