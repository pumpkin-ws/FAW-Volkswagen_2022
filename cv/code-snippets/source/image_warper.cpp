#include "image_warper.hpp"

using pt = spark_vision::ImageWarper;

void pt::calcMatrixFromPoints(
    const std::vector<cv::Point2f>& original_points, 
    const std::vector<cv::Point2f>& objective_points, 
    cv::Mat& homo_mat
) {
    if(original_points.size() == 0) {
        std::printf("No points in the original pixel data.\n");
        this->m_has_matrix = false;
        return;
    } else if(objective_points.size() == 0) {
        std::printf("No points in the objective pixel data.\n");
        this->m_has_matrix = false;
        return;
    } else if(original_points.size() != objective_points.size()) {
        std::cout << original_points.size() << std::endl;
        std::cout << objective_points.size() << std::endl;
        std::printf("The objective points and object points have different dimensions, recheck inputs!\n");
        this->m_has_matrix = false;
        return;
    } else {
        // homoe_mat is a transformation from the original points to the objective points
        homo_mat = cv::findHomography(original_points, objective_points);
        m_homo_mat = homo_mat.clone();
        this->m_has_matrix = true;
        return;
    }
};

void pt::help() {
    printf("To use this function, ");
}

void pt::warpImage(const cv::Mat& input_image, cv::Mat& output_image) {
    if(m_homo_mat.empty()) {
        this->m_has_matrix = false;
        std::printf("The homography matrix is empty!\n");
        return;
    } else if(this->m_has_matrix == false) {
        std::printf("Need to first calculate the homography matrix!\n");
        return;
    } else if(input_image.empty()){
        std::printf("The input image is empty!\n");
        return;
    } else if(input_image.size() != this->m_image_dims) {
        
        std::printf("The input dimension does not match the preset image dimension. Perspective warp will not work!\n");
        std::printf("Another possibility is the image dimensions are not set!\n");
        return;
    } else {
        cv::warpPerspective(input_image, output_image, this->m_homo_mat, input_image.size());
        return;
    }
}

