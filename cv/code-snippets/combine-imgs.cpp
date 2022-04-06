#include "opencv2/opencv.hpp"

/**
 * @brief Create a Strip object with vconcat
 * 
 */
void createStrip() {
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

int main(int argc, char** argv) {
    cv::Mat img1 = cv::imread("./data/1.bmp", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("./data/2.bmp", cv::IMREAD_COLOR);

    cv::resize(img1, img1, cv::Size(img1.size() / 4));
    cv::resize(img2, img2, cv::Size(img2.size() / 4));

    // TODO: the task is to find the vertical shift
    double vertical_shift{333.3 / 4};
    cv::Mat cropped_img2 = img2(cv::Rect(0, img2.rows - vertical_shift, img2.cols, vertical_shift)).clone();
    img2.release();
    // cv::line(cropped_img1, cv::Point(0, cropped_img1.rows/2), cv::Point(cropped_img1.cols, cropped_img1.rows/2), cv::Scalar(255, 0, 0), 5);

    cv::imshow("cropped", cropped_img2);

    cv::imshow("img1", img1);
    // cv::imshow("img2", img2);
    cv::Mat out;
    std::vector<cv::Mat> imgs{img1, cropped_img2};
    cv::vconcat(imgs, out);
    cv::imshow("combined", out);
    cv::waitKey(0);

    cv::imwrite("combined.jpg", out);

    

    return EXIT_SUCCESS;
   
}