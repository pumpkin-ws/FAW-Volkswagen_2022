#include "opencv2/opencv.hpp"

int main(int argc, char** argv) {
    cv::Mat img(cv::Size(300, 10), CV_8UC3, cv::Scalar::all(0));
    cv::Mat img2(cv::Size(300, 10), CV_8UC3, cv::Scalar::all(255));
    cv::imshow("img", img);
    cv::waitKey(0);
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < 100; i++) {
        imgs.push_back(img);
        imgs.push_back(img2);
    }
    cv::Mat out;
    cv::vconcat(imgs, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    cv::imwrite("stripes.jpg", out);
}