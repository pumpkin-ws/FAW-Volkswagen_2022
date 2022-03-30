#ifndef TARGET_DETECTOR_HPP_
#define TARGET_DETECTOR_HPP_

#include "opencv2/opencv.hpp"

/**
 * @brief findAndDrawSymmCircles - find and draw center points on symmetric circles
 * 
 * @param input 
 * @param tracked_centers output stores the tracked center calculation results
 * @param grid_size 
 * @param draw_result 
 * @return int 
 */
int findAndDrawSymmCircles(
    const cv::Mat& input, 
    std::vector<cv::Point2f>& tracked_centers, 
    const cv::Size& grid_size, 
    bool draw_result = true
);

#endif