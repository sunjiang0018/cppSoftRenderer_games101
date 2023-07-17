//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H

#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv4/opencv2/opencv.hpp>

class Texture {
private:
    cv::Mat image_data;

public:
    Texture(const std::string &name) {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v) {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v) {
        auto u_img = u * width;
        auto v_img = (1.0f - v) * height;
        auto u_min = std::floor(u_img);
        auto u_max = std::min(std::ceil(u_img), static_cast<float>(width));
        auto v_min = std::floor(v_img);
        auto v_max = std::min(std::ceil(v_img), static_cast<float>(height));

        auto t = (u_img - u_min) / (u_max - u_min) ;
        auto s = (v_img - v_max) / (v_min - v_max);

        auto color1 = image_data.at<cv::Vec3b>(v_max, u_min);
        auto color2 = image_data.at<cv::Vec3b>(v_max, u_max);
        auto color3 = image_data.at<cv::Vec3b>(v_min, u_min);
        auto color4 = image_data.at<cv::Vec3b>(v_min,  u_max);

        auto trep1 = color2 * t + (1.0f - t) * color1;
        auto trep2 = color4 * t + (1.0f - t) * color3;
        auto result = trep2 * s + (1.0f - s) * trep1;

        return Eigen::Vector3f(result[0], result[1], result[2]);
    }

};

#endif //RASTERIZER_TEXTURE_H
