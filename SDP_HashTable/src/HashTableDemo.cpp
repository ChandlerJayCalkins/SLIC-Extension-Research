// Demo.cpp
// Demonstrates SLIC
// Author: Chandler Calkins

#include "SLICHashTable.hpp"
#include "sdp_slic.hpp"
#include <iostream>
#include <string>
#ifdef _WIN32
    #include <direct.h>
    #define chdir _chdir
#else
    #include <unistd.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/ximgproc/slic.hpp>
#include <map>
using namespace cv;

// main - Generates superpixels for an images using SLIC and displays those superpixels on the image.
//
// Preconditions:
//
// There is a valid .jpg or .png file in the project folder and the `imread()` call that creates the `input_image` object reads from that file.
//
// Postconditions:
//
// A file called output.jpg should be in the project folder.
int main(int argc, char* argv[])
{
	int input_count = 4;
	// naming scheme: input0, input1,...inputN
	// loop through and hash each one

	// Move out of build/Debug into root of project folder
	// Use this for VSCode, comment out for Visual Studio / actual submission
	chdir("../../");

	std::vector<Mat> database_images(input_count);
	std::string ext = ".jpg";
	std::string base = "input";
	const int min_superpixel_size_percent = 4;
	const int avg_superpixel_size = 25; // Default: 100
	const float smoothness = 0.0f; // Default: 10.0

	// initialize hash table
	SLICHashTable hash_table;

	for (int i = 0; i < input_count; i++) {
		std::string num = std::to_string(i);
		// Reads the input image
		std::string file_name = base + num + ext;
		//std::cout << file_name << "\n" << std::endl; 
		database_images[i] = imread(file_name);
		Mat lab_image;
		//cvtColor(database_images[i], lab_image, COLOR_BGR2Lab);

		
		
		//Ptr<ximgproc::SuperpixelSLIC> slic = ximgproc::createSuperpixelSLIC(database_images[i], ximgproc::SLIC, avg_superpixel_size, smoothness);
		
		// Duperize
		Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(database_images[i], SLIC, avg_superpixel_size, smoothness);
		slic->iterate();
		slic->enforceLabelConnectivity(min_superpixel_size_percent);
		slic->duperizeWithAverage(25.0f);

		// Gets 2D array of the superpixel each pixel is a part of
		Mat labels;
		slic->getLabels(labels);
		int superpixel_count = slic->getNumberOfSuperpixels();

		// Counts how many pixels are in each superpixel
		unsigned long* pixel_count = (unsigned long*) calloc(superpixel_count, sizeof(unsigned long));
		for (int row = 0; row < labels.rows; row += 1)
		{
			for (int col = 0; col < labels.cols; col += 1)
			{
				pixel_count[labels.at<int>(row, col)] += 1;
			}
		}

		// hash superpixels in table
		hash_table.Hash(database_images[i], labels, superpixel_count, pixel_count);

		// Prints out the pixel count of each superpixel
		// for (int i = 0; i < superpixel_count; i += 1)
		// {
		// 	std::cout << i << ": " << pixel_count[i] << std::endl;
		// }

		// Gets overlay image of superpixels
		// Mat superpixels;
		// slic->getLabelContourMask(superpixels);

		// // Creates the output image of superpixels
		// Mat output(input_image);
		// // Set each pixel in output to white if it's a superpixel border
		// for (int row = 0; row < output.rows; row += 1)
		// for (int col = 0; col < output.cols; col += 1)
		// {
		// 	if (superpixels.at<uchar>(row, col) != 0)
		// 	{
		// 		output.at<Vec3b>(row, col)[0] = superpixels.at<uchar>(row, col);
		// 		output.at<Vec3b>(row, col)[1] = superpixels.at<uchar>(row, col);
		// 		output.at<Vec3b>(row, col)[2] = superpixels.at<uchar>(row, col);
		// 	}
		// }

		// // Displays output to a window
		// imshow(window_name, output);
		// waitKey(0);
		
		// // Write output to an image file
		// imwrite("output.png", output);
	}
	std::string qbase = "query";
	int q_count = 4;

	for (int i = 0; i < q_count; i++) {
		std::string num = std::to_string(i);
		std::string q_file_name = qbase + num + ext;
		// load query image
		Mat query_image = imread(q_file_name);

		// generate superpixels for query image
		Ptr<SuperpixelSLIC> query_slic = createSuperpixelSLIC(database_images[i], SLIC, avg_superpixel_size, smoothness);
		query_slic->iterate();
		query_slic->enforceLabelConnectivity(min_superpixel_size_percent);
		query_slic->duperizeWithAverage(25.0f);

		Mat query_labels;
		query_slic->getLabels(query_labels);
		int query_superpixel_count = query_slic->getNumberOfSuperpixels();

		// count pixels in each query superpixel
		unsigned long* query_pixel_count = (unsigned long*) calloc(query_superpixel_count, sizeof(unsigned long));
		for (int row = 0; row < query_labels.rows; row += 1) {
			for (int col = 0; col < query_labels.cols; col += 1) {
				query_pixel_count[query_labels.at<int>(row, col)] += 1;
			}
		}

		// build HashKey structs for query superpixels
		HashKey *query_superpixels = (HashKey*) calloc(query_superpixel_count, sizeof(HashKey));
		for (int row = 0; row < query_labels.rows; row++) {
			for (int col = 0; col < query_labels.cols; col++) {
				cv::Vec3b lab_pixel = query_image.at<cv::Vec3b>(row, col); // using BGR as LAB
				int sp = query_labels.at<int>(row, col);
				HashKey &curr = query_superpixels[sp];

				// add to total color values
				curr.l_tot += lab_pixel[0];
				curr.a_tot += lab_pixel[1];
				curr.b_tot += lab_pixel[2];

				// set initial params
				if (curr.pixel_count == 0) {
					curr.x_range.first = col;
					curr.x_range.second = col;
					curr.y_range.first = row;
					curr.y_range.second = row;
					curr.original_image = &query_image;
				// update spatial extent
				} else {
					if (curr.x_range.first > col) curr.x_range.first = col;
					if (curr.x_range.second < col) curr.x_range.second = col;
					if (curr.y_range.first > row) curr.y_range.first = row;
					if (curr.y_range.second < row) curr.y_range.second = row;
				}
				curr.pixel_count += 1;
			}
		}

		// find matches by counting hash collisions
		std::map<const cv::Mat*, int> match_counts;
		for (int i = 0; i < query_superpixel_count; i++) {
			int query_key = hash_table.calculate_hash_key(query_superpixels[i]);
			if (query_key == -1) continue;

			// check if this hash key exists in the table
			if (hash_table.hashTable.count(query_key)) {
				// if so, iterate all superpixels that share this key
				std::vector<HashKey>& matches = hash_table.hashTable[query_key];
				for (const HashKey& match : matches) {
					// increment the count for the image this superpixel belongs to
					match_counts[match.original_image]++;
				}
			}
		}

		// find the image with the highest match count
		const cv::Mat* best_match_image_ptr = nullptr;
		int max_matches = 0;
		for (auto const& pair : match_counts) {
			if (pair.second > max_matches) {
				max_matches = pair.second;
				best_match_image_ptr = pair.first;
			}
		}

		// display the query image and the best match
		namedWindow("Query Image");
		imshow("Query Image", query_image);

		if (best_match_image_ptr != nullptr) {
			namedWindow("Best Match");
			imshow("Best Match", *best_match_image_ptr);
		} else {
			std::cout << "No matches found." << std::endl;
		}
		
		waitKey(0);

		free(query_pixel_count);
		free(query_superpixels);
	}
	

	

	return 0;
}
