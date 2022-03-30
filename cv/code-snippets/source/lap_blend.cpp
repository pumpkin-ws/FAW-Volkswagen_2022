
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
    lapPyr.clear();
    cv::Mat currentImg = img;
    for (int l = 0; l < levels; l++) {
        cv::Mat down, up;
        cv::pyrDown(currentImg, down);
        cv::pyrUp(down, up, currentImg.size());
        cv::Mat lap = currentImg - up;
        lapPyr.push_back(lap);
        currentImg = down;
    }
    currentImg.copyTo(HighestLevel);
}

cv::Mat_<cv::Vec3f> LaplacianBlending::reconstructImgFromLapPyramid() {
    cv::Mat currentImg = resultHighestLevel;
    for(int l = levels - 1; l >=0; l--) {
        cv::Mat up;
        cv::pyrUp(currentImg, up, resultLapPyr[l].size());
        currentImg = up + resultLapPyr[l];
    }
    return currentImg;
}

void LaplacianBlending::blendLapPyrs() {
    resultHighestLevel = leftHighestLevel.mul(maskGaussianPyramid.back()) + 
        rightHighestLevel.mul(cv::Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());
    for (int l = 0; l < levels; l++) {
        cv::Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
        cv::Mat antiMask = cv::Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
        cv::Mat B = rightLapPyr[l].mul(antiMask);
        cv::Mat_<cv::Vec3f> blendedLevel = A + B;
        resultLapPyr.push_back(blendedLevel);
    }
}

