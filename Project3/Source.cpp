

#include <string>
#include <exception>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <dmtx.h>
using namespace std;
using namespace cv;


void test(cv::Mat img) 
{

    DmtxImage* dImg = dmtxImageCreate(img.data, img.cols, img.rows, DmtxPack8bppK); //DmtxPack24bppRGB
    DmtxDecode* dDec = dmtxDecodeCreate(dImg, 1);
    std::string qrcode;
    cv::Mat outImg;
    cv::cvtColor(img, outImg, cv::COLOR_GRAY2BGR);
    DmtxTime timeout;
    timeout.sec = 4;
    timeout.usec = timeout.sec*1000000;
    DmtxRegion* dReg = dmtxRegionFindNext(dDec, NULL);
    if (dReg != NULL)
    {
        DmtxMessage* dMsg = dmtxDecodeMatrixRegion(dDec, dReg, DmtxUndefined);
        if (dMsg!= NULL)
        {
            qrcode = std::string((char*)dMsg->output);
            std::cout << qrcode << std::endl;
        }
      
        int height = dmtxDecodeGetProp(dDec, DmtxPropHeight);
        DmtxVector2 topLeft, topRight, bottomLeft, bottomRight;
        topLeft.X = 0;
        topLeft.Y = 0;

        topRight.X = 1.0;
        topRight.Y = 0;

        bottomRight.X = 1.0;
        bottomRight.Y = 1.0;

        bottomLeft.X = 0;
        bottomLeft.Y = 1.0;
     
        dmtxMatrix3VMultiplyBy(&topLeft, dReg->fit2raw);
        dmtxMatrix3VMultiplyBy(&topRight, dReg->fit2raw);
        dmtxMatrix3VMultiplyBy(&bottomLeft, dReg->fit2raw);
        dmtxMatrix3VMultiplyBy(&bottomRight, dReg->fit2raw);

        double rotate = (2 * M_PI) + atan2(topLeft.Y - bottomLeft.Y, bottomLeft.X - topLeft.X);
        int rotateInt = (int)(rotate * 180.0 / M_PI + 0.5);
        rotateInt = rotateInt >= 360 ? rotateInt - 360 : rotateInt;


        cv::Point p1(topLeft.X, height - 1 - topLeft.Y);
        cv::Point p2(topRight.X, height - 1 - topRight.Y);
        cv::Point p3(bottomRight.X, height - 1 - bottomRight.Y);
        cv::Point p4(bottomLeft.X, height - 1 - bottomLeft.Y);
        cv::line(outImg, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 0, 255), 2);
        cv::line(outImg, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), cv::Scalar(0, 0, 255), 2);
        cv::line(outImg, cv::Point(p3.x, p3.y), cv::Point(p4.x, p4.y), cv::Scalar(0, 0, 255), 2);
        cv::line(outImg, cv::Point(p4.x, p4.y), cv::Point(p1.x, p1.y), cv::Scalar(0, 0, 255), 2);
        cv::imshow("DataMatrix", outImg);
        cv::waitKey(0);
    }
   

    dmtxDecodeDestroy(&dDec);
    dmtxImageDestroy(&dImg);

}

int main(int argc, char** argv) {

    int deviceId = 0;
    int captureWidth = 640;
    int captureHeight = 480;
    bool multi = false;

    // Open video captire
    VideoCapture videoCapture(deviceId);
    
    // Check if webcam is open
    if (not videoCapture.isOpened())
    {
        return -1;
    }


    // The captured image and its grey conversion
    Mat image, grey;


    // Stopped flag will be set to -1 from subsequent wayKey() if no key was pressed
    int stopped = -1;

  
    // Capture image
    image = cv::imread("C:/Users/dajo/source/repos/Project3/x64/Debug/datamatrix100.png", cv::IMREAD_GRAYSCALE); //videoCapture.read(image);
    test(image);
    while (videoCapture.isOpened())
    {
        videoCapture.read(image);
        if (image.empty())
            break;
        // Show captured image
        cv::imshow("DataMatrix", image);
        cv::waitKey(10);
        cv::Mat decodeMat;
        if (image.channels() == 3)
        {

            cv::cvtColor(image, decodeMat, cv::COLOR_BGR2GRAY);
        }
        test(decodeMat);
    }
   

    // fixImage(image, img);
    //detector(image, img);
    // if(img != NULL)
    // decoder(img);
           

    // Release video capture
    videoCapture.release();

    return 0;

}
