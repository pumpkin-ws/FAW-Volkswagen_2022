#include "opencv2/opencv.hpp"
#include "source/file_manip.hpp"
#include "source/image_warper.hpp"
#include "source/target_detector.hpp"
#include "source/lap_blend.hpp"
#include <algorithm>

cv::Mat_<cv::Vec3f> LaplacianBlend(const cv::Mat_<cv::Vec3f>& l, const cv::Mat_<cv::Vec3f>& r, const cv::Mat_<float>& m) {
    LaplacianBlending lb(l, r, m, 2);
    return lb.blend();
}

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

    for (int i = 0; i < img_fnames.size(); i++) {
        std::string save_name = "./data/movement-result/resize" + std::to_string(i) + ".jpg";
        cv::imwrite(save_name, input_imgs[i]);
    }

    /* detect the keypoints the image */
    std::vector<cv::Point2f> centers;
    findAndDrawSymmCircles(input_imgs[0], centers, cv::Size(7, 7), true);

    /* unwarp image */
    spark_vision::ImageWarper warper;

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
    cv::waitKey(50);

    cv::Mat concat_compare;
    std::vector<cv::Mat> h_imgs{input_imgs[0], out};
    cv::vconcat(h_imgs, concat_compare);
    cv::imshow("hconcat", concat_compare);
    cv::waitKey(50);

    std::vector<cv::Mat> unwarped_imgs;
    cv::destroyAllWindows();
    for (int i = 0; i < input_imgs.size(); i++) {
        cv::Mat unwarp;
        warper.warpImage(input_imgs[i], unwarp);
        unwarped_imgs.push_back(unwarp);
        std::string save_name = "./data/movement-result/unwarp" + std::to_string(i) + ".jpg";
        cv::imwrite(save_name, unwarp);
        std::vector<cv::Mat> vert{input_imgs[i], unwarp};
        cv::Mat concat;
        
        cv::hconcat(vert, concat);

        cv::imshow("unwarp concat", concat);
        cv::imshow("input", input_imgs[i]);
        cv::imshow("uunwarp", unwarp);
        cv::waitKey(50);
    }
    cv::destroyAllWindows();

    std::vector<cv::Point2f> centers0;
    std::vector<cv::Point2f> centers1;
    findAndDrawSymmCircles(input_imgs[0], centers0, cv::Size(7, 7), false);
    findAndDrawSymmCircles(input_imgs[1], centers1, cv::Size(7, 7), false);

    printf("The coordinate of the 0th point of centers0: \n");
    std::cout << centers0[0] << std::endl;
    printf("The coordinate of the 0th point of centers1: \n");
    std::cout << centers1[0] << std::endl;

    double pixel_to_distance = std::fabs((centers0[0].y - centers1[0].y)) / 15;
    std::cout << "The pixel to distance is: " << pixel_to_distance << std::endl;
    
    /* blending with laplacian */
    cv::Mat total_mat = unwarped_imgs[0].clone();
    for (int i = 1; i < unwarped_imgs.size(); i++) {
        
        cv::Mat clipped_mat = unwarped_imgs[i](cv::Rect(0, unwarped_imgs[i].rows - int(pixel_to_distance * 15), unwarped_imgs[i].cols, int(pixel_to_distance * 15))).clone();
        std::vector<cv::Mat> combine_imgs{total_mat, clipped_mat};

        cv::Mat_<float> mask(clipped_mat.rows + total_mat.rows, total_mat.cols, 0.0);
        mask(cv::Range(total_mat.rows, total_mat.rows + clipped_mat.rows), cv::Range(0, total_mat.cols)) = 1.0;
        cv::imshow("mask", mask);

        cv::Mat_<cv::Vec3f> top;
        if (i == 1) {
            total_mat.convertTo(top, CV_32F, 1.0/255.0);
        } else {
            top = total_mat.clone();
        }
        cv::Mat_<cv::Vec3f> top_append(clipped_mat.rows, clipped_mat.cols, 0.0);
        top.push_back(top_append);
        cv::imshow("top", top);

        cv::Mat_<cv::Vec3f> bottom(total_mat.rows, total_mat.cols, 0.0);
        cv::Mat_<cv::Vec3f> clipped_bot;
        clipped_mat.convertTo(clipped_bot, CV_32F, 1.0/255.0);
        bottom.push_back(clipped_bot);
        cv::imshow("bottom", bottom);
        // cv::waitKey(0);



        total_mat = LaplacianBlend(bottom, top, mask);
        cv::imshow("blend", total_mat);
    }
    cv::destroyAllWindows();
    cv::imshow("laplacian combine", total_mat);
    cv::waitKey(0);

    /* simple concatenation */
    std::vector<cv::Mat> unwarped_gray;
    for (int i = 0; i < unwarped_imgs.size(); i++) {
        cv::Mat gray;
        cv::cvtColor(unwarped_imgs[i], gray, cv::COLOR_BGR2GRAY);
        unwarped_gray.push_back(gray);
    }
    cv::Mat total_mat1 = unwarped_gray[0].clone();
    for (int i = 1; i < unwarped_gray.size(); i++) {
        cv::Mat clipped_mat = unwarped_gray[i](cv::Rect(0, unwarped_gray[i].rows - int(pixel_to_distance * 15), unwarped_gray[i].cols, int(pixel_to_distance * 15)));
        std::vector<cv::Mat> combine_imgs{total_mat1, clipped_mat};
        cv::vconcat(combine_imgs, total_mat1);
    }
    cv::imshow("simple combine", total_mat1);
    cv::waitKey(0);

    cv::imwrite("./data/movement-result/simple_concat.jpg", total_mat1);

    /* gray readjusted */
    if (unwarped_imgs[0].type() == CV_8UC1) {
        printf("The type is 8UC1\n");
    } else if (unwarped_imgs[0].type() == CV_8UC3) {
        printf("The type is 8UC3\n");
    }
    cv::destroyAllWindows();


    cv::Mat total_mat2(unwarped_gray[0].size(), CV_64FC1, cv::Scalar(0));
    for (int i = 0; i < total_mat2.rows; i++) {
        for (int j = 0; j < total_mat2.cols; j++) {
            total_mat2.at<double>(i, j) = (double)(unwarped_gray[0].at<uchar>(i, j)) / 1000.0;
            if (fabs((double)(unwarped_gray[0].at<uchar>(i, j))) < 1e-5) {
                total_mat2.at<double>(i, j) = 1.0 / 1000.0;
            }
        }
    }
    for (int i = 1; i < unwarped_gray.size(); i++) {
        cv::Mat clipped_mat_uchar = unwarped_gray[i](cv::Rect(0, unwarped_gray[i].rows - int(pixel_to_distance * 15), unwarped_gray[i].cols, int(pixel_to_distance * 15))).clone();
        cv::Mat clipped_mat_double(clipped_mat_uchar.size(), CV_64FC1, cv::Scalar(0));
        for (int j = 0; j < clipped_mat_double.rows; j++) {
            for (int k = 0; k < clipped_mat_double.cols; k++) {
                clipped_mat_double.at<double>(j, k) = (double)((double)clipped_mat_uchar.at<uchar>(j, k) / 1000.0);
                if (fabs((double)clipped_mat_uchar.at<uchar>(j, k)) < 1e-5) {
                    clipped_mat_double.at<double>(j, k) = 1.0 / 1000.0;
                }
            }
        }
        for (int j = 0; j < clipped_mat_double.cols; j++) {
            std::cout << total_mat2.at<double>(total_mat2.rows - 1, j) << std::endl;
            std::cout << clipped_mat_double.at<double>(0, j) << std::endl;
            double adjust_scale = total_mat2.at<double>(total_mat2.rows - 1, j) / clipped_mat_double.at<double>(0, j);
            for (int k = 0; k < clipped_mat_double.rows; k++) {
                clipped_mat_double.at<double>(k, j) *= adjust_scale;
            }
        }
        cv::GaussianBlur(clipped_mat_double, clipped_mat_double, cv::Size(3, 3), 1, 3);
        std::vector<cv::Mat> combine_imgs{total_mat2, clipped_mat_double};
        cv::vconcat(combine_imgs, total_mat2);
        cv::imshow("combine", total_mat2);
        cv::waitKey(50);
    }

    for (int i = 0; i < total_mat2.rows; i++) {
        for (int j = 0; j < total_mat2.cols; j++) {
            total_mat2.at<double>(i, j) *= 1000;
        }
    }
    cv::imshow("combine", total_mat2);
    cv::waitKey(0);
    cv::Mat output;
    total_mat2.convertTo(output, CV_8UC1);

    // cv::equalizeHist(output, output);

    cv::imwrite("./data/movement-result/gray_stitched.jpg", output);

    

    
    return EXIT_SUCCESS;
}