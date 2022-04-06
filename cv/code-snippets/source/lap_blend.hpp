#ifndef LAP_BLEND_HPP_
#define LAP_BLEND_HPP_
#include <opencv2/opencv.hpp>

class LaplacianBlending {
private:
    cv::Mat_<cv::Vec3f> left;
    cv::Mat_<cv::Vec3f> right;
    cv::Mat_<float> blendMask;

    std::vector<cv::Mat_<cv::Vec3f>> leftLapPyr, rightLapPyr, resultLapPyr; // Laplacian Pyramids
    cv::Mat leftHighestLevel, rightHighestLevel, resultHighestLevel;
    std::vector<cv::Mat_<cv::Vec3f>> maskGaussianPyramid; // masks are 3-channels for easier multiplication with RGB

    int levels;

    void buildPyramids();
    void buildGaussianPyramid();
    void buildLaplacianPyramid(const cv::Mat& img, std::vector<cv::Mat_<cv::Vec3f>>& lapPyr, cv::Mat& HighestLevel);
    cv::Mat_<cv::Vec3f> reconstructImgFromLapPyramid();
    void blendLapPyrs();

public:
    LaplacianBlending(const cv::Mat_<cv::Vec3f>& _left, const cv::Mat_<cv::Vec3f>& _right, const cv::Mat_<float>& _blendMask, int _levels)
    : left(_left), right(_right), blendMask(_blendMask), levels(_levels) {
        std::cout << "Image of the left side is " << _left.size() << std::endl;
        std::cout << "Image of the right side is " <<_right.size() << std::endl;
        assert(_left.size() == _right.size()); // why left size needs to be equal to right size?
        assert(_left.size() == _blendMask.size());
        buildPyramids(); // construct laplacian pyramid and gaussian pyramid
        blendLapPyrs(); // blend left and right pyramids into one pyramid
    };  

    cv::Mat_<cv::Vec3f> blend() {
        return reconstructImgFromLapPyramid();
    }

};

#endif