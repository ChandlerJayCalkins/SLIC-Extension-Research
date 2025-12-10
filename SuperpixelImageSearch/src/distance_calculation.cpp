#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

/* ---------- utility ---------- */

cv::Mat load_gray(const fs::path& p)
{
    cv::Mat img = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty())
        throw std::runtime_error("Could not load image: " + p.string());
    return img;
}

cv::Mat sift_descriptor(const cv::Mat& img)
{
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    sift->detectAndCompute(img, cv::noArray(), kps, desc);

    if (desc.empty())
        return cv::Mat::zeros(1, 128, CV_32F);

    // Mean descriptor (GLOBAL-style aggregation)
    cv::Mat mean;
    cv::reduce(desc, mean, 0, cv::REDUCE_AVG, CV_32F);
    return mean;
}

float l2_distance(const cv::Mat& a, const cv::Mat& b)
{
    return static_cast<float>(cv::norm(a, b, cv::NORM_L2));
}

/* ---------- main ---------- */

int main()
{
    try
    {
        fs::path output_root = "SuperpixelImageSearch/output";
        fs::path query_path  = output_root / "origin" / "query_original.jpg";
        fs::path csv_out     = output_root / "csv" / "distance_posthoc.csv";

        cv::Mat query_img  = load_gray(query_path);
        cv::Mat query_desc = sift_descriptor(query_img);

        std::ofstream csv(csv_out);
        csv << "method,match_index,distance\n";

        for (const auto& dir : fs::directory_iterator(output_root))
        {
            if (!dir.is_directory()) continue;

            std::string method = dir.path().filename().string();
            if (method == "origin" || method == "csv") continue;

            for (int i = 1; i <= 5; ++i)
            {
                fs::path match_path = dir.path() / ("match_" + std::to_string(i) + ".jpg");
                if (!fs::exists(match_path)) continue;

                cv::Mat match_img  = load_gray(match_path);
                cv::Mat match_desc = sift_descriptor(match_img);

                float dist = l2_distance(query_desc, match_desc);
                csv << method << "," << i << "," << dist << "\n";
            }
        }

        csv.close();
        std::cout << "Saved distance CSV to: " << csv_out << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
