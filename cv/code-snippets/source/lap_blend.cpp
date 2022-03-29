
#include "lap_blend.hpp"

void LaplacianBlending::buildPyramids() {
    buildLaplacianPyramid(left, leftLapPyr, leftHighestLevel);
    buildLaplacianPyramid(right, rightLapPyr, rightHighestLevel);
    buildGaussianPyramid();
}

void LaplacianBlending::buildGaussianPyramid() {
    assert(leftLapPyr.size() > 0);
    maskGaussianPyramid.clear();
    cv::Mat currentImg;
    cv::cvtColor(blendMask, currentImg, cv::COLOR_GRAY2BGR); // conversion from gray to color
    maskGaussianPyramid.push_back(currentImg); // 0-level

    currentImg = blendMask;
    for (int l = 1; l < levels + 1; l++) {
        cv::Mat _down;
        if (leftLapPyr.size() > l) {
            cv::pyrDown(currentImg, _down, leftLapPyr[l].size());
        } else {
            pyrDown(currentImg, _down, leftHighestLevel.size());
        }
        cv::Mat down;
        cv::cvtColor(_down, down, cv::COLOR_GRAY2BGR);
        maskGaussianPyramid.push_back(down);
        currentImg = _down;
    }

}

void LaplacianBlending::buildLaplacianPyramid(const cv::Mat& img, std::vector<cv::Mat_<cv::Vec3f>>& lapPyr, cv::Mat& HighestLevel) {
    
}