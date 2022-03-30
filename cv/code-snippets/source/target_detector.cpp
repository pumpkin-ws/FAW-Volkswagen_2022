#include "target_detector.hpp"


int findAndDrawSymmCircles(
    const cv::Mat& input, 
    std::vector<cv::Point2f>& tracked_centers, 
    const cv::Size& grid_size, 
    bool draw_result) {

    cv::Mat track_result;
    cv::Mat display_mat = input.clone();
    tracked_centers.clear();
    if(cv::findCirclesGrid(input, grid_size, track_result)) {
        // if track is successful
        for (int i = 0; i < track_result.rows; i++) {
            tracked_centers.push_back(track_result.at<cv::Point2f>(0, i));
            if (draw_result == true) {
                cv::circle(display_mat, tracked_centers[i], 10, cv::Scalar(0, 0, 255), 5, cv::FILLED);
                cv::putText(display_mat, std::to_string(i), tracked_centers[i], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255));
            }
        }
        if (draw_result == true) {
            cv::imshow("found circles", display_mat);
            cv::waitKey(0);
        }
        return 0;
    } else {
        // if find circle grid is unsuccessful, return unsuccessful code of 1
        return 1;   
    };

};