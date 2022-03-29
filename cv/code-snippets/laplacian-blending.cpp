#include "source/lap_blend.hpp"

cv::Mat_<cv::Vec3f> LaplacianBlend(const cv::Mat_<cv::Vec3f>& l, const cv::Mat_<cv::Vec3f>& r, const cv::Mat_<float>& m) {
    LaplacianBlending lb(l, r, m, 4);
    return lb.blend();
}

int main(int argc, char** argv) {
    return EXIT_SUCCESS;
}   