/**
 * @file test_complete_pipeline.cpp
 * @brief End-to-end test of the complete LTriDP superpixel segmentation pipeline
 * 
 * @author Ketsia Mbaku
 * 
 * This program demonstrates the complete pipeline:
 * 1. Preprocessing (3D histogram reconstruction + gamma transformation)
 * 2. Feature Extraction (LTriDP texture descriptor)
 * 3. Superpixel Segmentation (LTriDP-enhanced SLIC)
 * 
 * Usage: ./test_complete_pipeline ../data/input ../data/output
 */

#include "preprocessing.hpp"
#include "feature_extraction.hpp"
#include "slic.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

/**
 * @brief Generate visualization of superpixel segmentation
 * 
 * Creates a colorized visualization where each superpixel is filled
 * with a unique color, with boundaries overlaid.
 */
cv::Mat visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels, const cv::Mat& boundaries) {
    // Convert grayscale to BGR for colorization
    cv::Mat viz;
    cv::cvtColor(image, viz, cv::COLOR_GRAY2BGR);
    
    // Create color map for each superpixel
    int num_superpixels = 0;
    cv::minMaxLoc(labels, nullptr, reinterpret_cast<double*>(&num_superpixels));
    num_superpixels++;
    
    std::vector<cv::Vec3b> colors(num_superpixels);
    for (int i = 0; i < num_superpixels; ++i) {
        colors[i] = cv::Vec3b(
            static_cast<uchar>((i * 73) % 256),
            static_cast<uchar>((i * 137) % 256),
            static_cast<uchar>((i * 211) % 256)
        );
    }
    
    // Apply colors with 50% transparency
    for (int y = 0; y < viz.rows; ++y) {
        for (int x = 0; x < viz.cols; ++x) {
            int label = labels.at<int>(y, x);
            cv::Vec3b pixel = viz.at<cv::Vec3b>(y, x);
            cv::Vec3b color = colors[label];
            viz.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (pixel[0] + color[0]) / 2,
                (pixel[1] + color[1]) / 2,
                (pixel[2] + color[2]) / 2
            );
        }
    }
    
    // Overlay boundaries in white
    for (int y = 0; y < viz.rows; ++y) {
        for (int x = 0; x < viz.cols; ++x) {
            if (boundaries.at<uchar>(y, x) == 255) {
                viz.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    
    return viz;
}

/**
 * @brief Generate comparison grid showing all pipeline stages
 */
cv::Mat createComparisonGrid(const cv::Mat& original,
                              const cv::Mat& enhanced,
                              const cv::Mat& features,
                              const cv::Mat& superpixels) {
    // Resize all to same size if needed
    cv::Size target_size(original.cols, original.rows);
    
    cv::Mat enhanced_display, features_display;
    if (enhanced.size() != target_size) {
        cv::resize(enhanced, enhanced_display, target_size);
    } else {
        enhanced_display = enhanced.clone();
    }
    
    if (features.size() != target_size) {
        cv::resize(features, features_display, target_size);
    } else {
        features_display = features.clone();
    }
    
    // Convert grayscale images to BGR for uniform grid
    cv::Mat original_bgr, enhanced_bgr, features_bgr;
    cv::cvtColor(original, original_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(enhanced_display, enhanced_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(features_display, features_bgr, cv::COLOR_GRAY2BGR);
    
    // Add labels to each image
    cv::putText(original_bgr, "1. Original", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(enhanced_bgr, "2. Enhanced", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(features_bgr, "3. LTriDP", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(superpixels, "4. Superpixels", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    // Create 2×2 grid
    cv::Mat top_row, bottom_row, grid;
    cv::hconcat(original_bgr, enhanced_bgr, top_row);
    cv::hconcat(features_bgr, superpixels, bottom_row);
    cv::vconcat(top_row, bottom_row, grid);
    
    return grid;
}

/**
 * @brief Process a single MRI image through the complete pipeline
 */
bool processImage(const fs::path& input_path, const fs::path& output_dir) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Processing: " << input_path.filename() << "\n";
    std::cout << std::string(80, '=') << "\n";
    
    // ========================================================================
    // Step 1: Load image
    // ========================================================================
    cv::Mat original = cv::imread(input_path.string(), cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Error: Could not load image: " << input_path << "\n";
        return false;
    }
    
    std::cout << "✓ Loaded image: " << original.cols << "×" << original.rows << " pixels\n";
    
    // ========================================================================
    // Step 2: Preprocessing (3D histogram + gamma enhancement)
    // ========================================================================
    std::cout << "\nStep 1: Preprocessing...\n";
    ltridp_slic_improved::Preprocessor preprocessor;
    cv::Mat enhanced;
    preprocessor.enhance(original, enhanced, 0.5f);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(enhanced, mean, stddev);
    std::cout << "  Enhanced image statistics:\n";
    std::cout << "    Mean: " << std::fixed << std::setprecision(2) << mean[0] << "\n";
    std::cout << "    StdDev: " << stddev[0] << "\n";
    
    // ========================================================================
    // Step 3: Feature Extraction (LTriDP texture descriptor)
    // ========================================================================
    std::cout << "\nStep 2: Feature Extraction (LTriDP)...\n";
    ltridp_slic_improved::FeatureExtractor feature_extractor;
    cv::Mat features;
    feature_extractor.extract(enhanced, features);
    
    cv::meanStdDev(features, mean, stddev);
    std::cout << "  Feature map statistics:\n";
    std::cout << "    Mean: " << mean[0] << "\n";
    std::cout << "    StdDev: " << stddev[0] << "\n";
    
    // ========================================================================
    // Step 4: Superpixel Segmentation (LTriDP-enhanced SLIC)
    // ========================================================================
    std::cout << "\nStep 3: Superpixel Segmentation (LTriDP SLIC)...\n";
    
    // Try different region sizes
    std::vector<int> region_sizes = {10, 20, 30};
    
    for (int region_size : region_sizes) {
        std::cout << "\n  Region size: " << region_size << " pixels\n";
        
        ltridp::LTriDPSuperpixelSLIC slic(enhanced, features, region_size, 10.0f);
        
        // Run 10 iterations
        slic.iterate(10);
        
        int num_superpixels = slic.getNumberOfSuperpixels();
        std::cout << "    Number of superpixels: " << num_superpixels << "\n";
        
        // Get labels and boundaries
        cv::Mat labels, boundaries;
        slic.getLabels(labels);
        slic.getLabelContourMask(boundaries);
        
        // Enforce connectivity
        slic.enforceLabelConnectivity(25);
        int final_superpixels = slic.getNumberOfSuperpixels();
        std::cout << "    After connectivity: " << final_superpixels << " superpixels\n";
        
        // Get updated labels after connectivity
        slic.getLabels(labels);
        
        // Create visualizations
        cv::Mat superpixel_viz = visualizeSuperpixels(enhanced, labels, boundaries);
        cv::Mat comparison_grid = createComparisonGrid(original, enhanced, features, superpixel_viz);
        
        // Calculate superpixel statistics
        int boundary_pixels = cv::countNonZero(boundaries);
        float boundary_percentage = 100.0f * static_cast<float>(boundary_pixels) / 
                                   static_cast<float>(enhanced.rows * enhanced.cols);
        std::cout << "    Boundary pixels: " << boundary_pixels 
                  << " (" << std::fixed << std::setprecision(2) 
                  << boundary_percentage << "%)\n";
        
        // Save outputs
        std::string base_name = input_path.stem().string();
        std::string size_suffix = "_S" + std::to_string(region_size);
        
        fs::path labels_path = output_dir / (base_name + size_suffix + "_labels.png");
        fs::path boundaries_path = output_dir / (base_name + size_suffix + "_boundaries.png");
        fs::path viz_path = output_dir / (base_name + size_suffix + "_superpixels.png");
        fs::path grid_path = output_dir / (base_name + size_suffix + "_pipeline.png");
        
        // Convert labels to visualization
        cv::Mat labels_viz;
        labels.convertTo(labels_viz, CV_8U, 255.0 / final_superpixels);
        cv::applyColorMap(labels_viz, labels_viz, cv::COLORMAP_JET);
        
        cv::imwrite(labels_path.string(), labels_viz);
        cv::imwrite(boundaries_path.string(), boundaries);
        cv::imwrite(viz_path.string(), superpixel_viz);
        cv::imwrite(grid_path.string(), comparison_grid);
        
        std::cout << "    ✓ Saved: " << labels_path.filename() << "\n";
        std::cout << "    ✓ Saved: " << boundaries_path.filename() << "\n";
        std::cout << "    ✓ Saved: " << viz_path.filename() << "\n";
        std::cout << "    ✓ Saved: " << grid_path.filename() << "\n";
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   LTriDP Superpixel Segmentation - Complete Pipeline Test         ║\n";
    std::cout << "║   Preprocessing → Feature Extraction → Superpixel Clustering      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>\n";
        std::cerr << "\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " ../data/input ../data/output\n";
        std::cerr << "\n";
        return 1;
    }
    
    fs::path input_dir(argv[1]);
    fs::path output_dir(argv[2]);
    
    // Validate input directory
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << input_dir << "\n";
        return 1;
    }
    
    // Create output directory
    fs::create_directories(output_dir);
    std::cout << "Input directory:  " << fs::absolute(input_dir) << "\n";
    std::cout << "Output directory: " << fs::absolute(output_dir) << "\n";
    
    // Find all image files
    std::vector<fs::path> image_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
                ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                image_files.push_back(entry.path());
            }
        }
    }
    
    if (image_files.empty()) {
        std::cerr << "\nError: No image files found in " << input_dir << "\n";
        std::cerr << "Supported formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff\n";
        return 1;
    }
    
    std::cout << "\nFound " << image_files.size() << " image(s) to process\n";
    
    // Process each image
    int success_count = 0;
    int failure_count = 0;
    
    for (const auto& image_path : image_files) {
        if (processImage(image_path, output_dir)) {
            success_count++;
        } else {
            failure_count++;
        }
    }
    
    // Summary
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Processing Complete\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Successfully processed: " << success_count << " image(s)\n";
    if (failure_count > 0) {
        std::cout << "Failed: " << failure_count << " image(s)\n";
    }
    std::cout << "\nOutput files saved to: " << fs::absolute(output_dir) << "\n";
    std::cout << "\nGenerated files per image (for each region size S=10,20,30):\n";
    std::cout << "  *_S{N}_labels.png        - Colorized label map\n";
    std::cout << "  *_S{N}_boundaries.png    - Binary boundary mask\n";
    std::cout << "  *_S{N}_superpixels.png   - Superpixel visualization\n";
    std::cout << "  *_S{N}_pipeline.png      - Complete pipeline comparison grid\n";
    std::cout << "\n";
    
    return (failure_count == 0) ? 0 : 1;
}
