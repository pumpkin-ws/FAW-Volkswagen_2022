#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::Mat m = (cv::Mat_<uchar>(2, 2) << 1, 2, 3, 4);
    std::cout << m << std::endl;
    std::cout << "The value: " << std::endl;
    /* two ways of indexing are equivalent */
    std::cout << (double)m.at<uchar>(1, 0) << std::endl;
    std::cout << (double)m.at<uchar>(cv::Point(0, 1)) << std::endl;
    std::cout << std::boolalpha << (m.type() == CV_8UC1) << std::endl;
    return EXIT_SUCCESS;
}