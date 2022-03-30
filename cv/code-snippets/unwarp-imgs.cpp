#include "opencv2/opencv.hpp"
#include "source/file_manip.hpp"
#include "source/image_warper.hpp"
#include "source/target_detector.hpp"
#include <algorithm>

int main(int argc, char** argv) {
    std::vector<std::string> img_fnames = getAllFileName("./data/movement-src/", ".bmp");
    printf("Images contained in folder are: \n");
    std::sort(img_fnames.begin(), img_fnames.end(), [](std::string str1, std::string str2) {
        std::string num1 = str1.substr(0, str1.find_first_of(' '));
        std::string num2 = str2.substr(0, str2.find_first_of(' '));
        return atof(num1.c_str()) < atof(num2.c_str());
    });
    std::vector<cv::Mat> input_imgs;
    for (auto name : img_fnames) {
        std::cout << name << std::endl;
        cv::Mat img = cv::imread("./data/movement-src/" + name, cv::IMREAD_COLOR);
        cv::resize(img, img, img.size()/8);
        input_imgs.push_back(img);
    }
    /* detect the keypoints the image */
    std::vector<cv::Point2f> centers;
    findAndDrawSymmCircles(input_imgs[0], centers, cv::Size(7, 7), true);

    /* unwarp image */
    spark_vision::ImageWarper warper;
    // warper.help();

    std::vector<cv::Point2f> original_points;
    std::vector<cv::Point2f> objective_points;

    original_points.push_back(centers[0]);
    original_points.push_back(centers[6]);
    original_points.push_back(centers[42]);
    original_points.push_back(centers[48]);

    double horizontal_dist = centers[6].x - centers[0].x;
    objective_points.push_back(cv::Point2f(centers[0].x, centers[0].y));
    objective_points.push_back(cv::Point2f(centers[0].x + horizontal_dist, centers[0].y));
    objective_points.push_back(cv::Point2f(centers[0].x, centers[0].y + horizontal_dist));
    objective_points.push_back(cv::Point2f(centers[0].x + horizontal_dist, centers[0].y + horizontal_dist));

    std::cout << original_points.size() << std::endl;
    std::cout << objective_points.size() << std::endl;
    cv::Mat homo;
    warper.calcMatrixFromPoints(original_points, objective_points, homo);
    std::cout << "The homogeneous matrix is " << std::endl;
    std::cout << homo << std::endl;

    cv::Mat out;
    warper.set_image_dims(input_imgs[0].size());
    warper.warpImage(input_imgs[0], out);
    cv::imshow("unwarped image", out);
    cv::waitKey(0);

    cv::Mat concat_compare;
    std::vector<cv::Mat> h_imgs{input_imgs[0], out};
    cv::vconcat(h_imgs, concat_compare);
    cv::imshow("hconcat", concat_compare);
    cv::waitKey(0);

    std::vector<cv::Mat> unwarped_imgs;
    cv::destroyAllWindows();
    for (int i = 0; i < input_imgs.size(); i++) {
        cv::Mat unwarp;
        warper.warpImage(input_imgs[i], unwarp);
        unwarped_imgs.push_back(unwarp);
        std::vector<cv::Mat> vert{input_imgs[i], unwarp};
        cv::Mat concat;
        
        cv::hconcat(vert, concat);
        cv::line(unwarp, cv::Point2f(centers[0].x, 0), cv::Point2f(centers[0].x, unwarp.size().height), cv::Scalar(0, 0, 255), 3);
        cv::line(input_imgs[i], cv::Point2f(centers[0].x, 0), cv::Point2f(centers[0].x, unwarp.size().height), cv::Scalar(0, 0, 255), 3);

        cv::imshow("unwarp concat", concat);
        cv::imshow("input", input_imgs[i]);
        cv::imshow("uunwarp", unwarp);
        cv::waitKey();
    }
    
    return EXIT_SUCCESS;
}