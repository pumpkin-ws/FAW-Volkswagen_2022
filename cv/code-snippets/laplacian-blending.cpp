#include "source/lap_blend.hpp"

cv::Mat_<cv::Vec3f> LaplacianBlend(const cv::Mat_<cv::Vec3f>& l, const cv::Mat_<cv::Vec3f>& r, const cv::Mat_<float>& m) {
    LaplacianBlending lb(l, r, m, 10);
    return lb.blend();
}

int main(int argc, char** argv) {
    cv::Mat left = cv::imread("./data/1.bmp", cv::IMREAD_COLOR);
    cv::Mat right = cv::imread("./data/2.bmp", cv::IMREAD_COLOR);

    cv::resize(left, left, left.size() / 4, 0, 0, cv::INTER_LANCZOS4);
    cv::resize(right, right, right.size() / 4, 0, 0, cv::INTER_LANCZOS4);

    cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::waitKey(0);

    /* Perform merge */
    cv::Mat_<cv::Vec3f> l;
    left.convertTo(l, CV_32F, 1.0/255.0);
    cv::Mat_<cv::Vec3f> r;
    right.convertTo(r, CV_32F, 1.0/255.0);

    cv::destroyAllWindows();

    /* Create blend mask matrix m */
    std::cout << l.rows << ", " << l.cols << std::endl;
    cv::Mat_<float> m(l.rows, l.cols, 0.0);
    m(cv::Range(0, m.rows), cv::Range(0, m.cols / 2)) = 1.0;

    // cv::imshow("mask", m);
    
    cv::Mat_<cv::Vec3f> blend = LaplacianBlend(l, r, m);
    std::cout << "The number of channels in blend is " << blend.channels() << std::endl;
    cv::Mat result = blend.clone();
    cv::imshow("blended", result);
    cv::waitKey(0);
    return EXIT_SUCCESS;
}   