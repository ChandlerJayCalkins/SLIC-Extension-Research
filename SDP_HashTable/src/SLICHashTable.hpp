#ifndef SLICHASHTABLE_HPP
#define SLICHASHTABLE_HPP

#include <opencv2/core/mat.hpp>
#include <stdlib.h>
#include <unordered_map>

/* a HashKey pointer will point to an array of size superpixel_count (+1 if labels start at 1), 
in which HashKeys will occupy the indices of the superpixels they represent.
Each pixel will be looped through and the respective HashKey structs of the superpixels they belong
to will be dynamically updated.  */
typedef struct {
    signed long l_tot, a_tot, b_tot;
    std::pair<int, int> x_range, y_range;
    const cv::Mat *original_image;
    unsigned long pixel_count;
} HashKey;

/* Class containing an implementation of a hash map meant to hold copies of structs containing superpixel information 
   Also contains method for hashing (all) superpixels in a an image and an internal method for calculating hash keys */
class SLICHashTable {
    private:
        const int n = 5;
        // this implementation assumes 8-bit unsigned integer images
        const int lab_buckets = 16;
        const int lab_bucket_size = 256 / lab_buckets;
        const int max_img_w = 3840;
        const int max_img_h = 2160;
        const int x_buckets = 10;
        const int y_buckets = 10;
        const int x_bucket_size = max_img_w / x_buckets;
        const int y_bucket_size = max_img_h / y_buckets;
        int dims[5] = {lab_buckets, lab_buckets, lab_buckets, x_buckets, y_buckets};

        

    public:
        std::unordered_map<int, std::vector<HashKey>> hashTable;

        int calculate_hash_key(const HashKey& key) {
            if (key.pixel_count == 0) return -1;

            // calaculate average color values
            float l_avg = (float)key.l_tot / key.pixel_count;
            float a_avg = (float)key.a_tot / key.pixel_count;
            float b_avg = (float)key.b_tot / key.pixel_count;
            float x_center = (key.x_range.first + key.x_range.second) / 2.0f;
            float y_center = (key.y_range.first + key.y_range.second) / 2.0f;

            // calculate color buckets
            int l_bucket = (int)(l_avg / lab_bucket_size);
            int a_bucket = (int)(a_avg / lab_bucket_size);
            int b_bucket = (int)(b_avg / lab_bucket_size);
            int x_bucket = (int)(x_center / x_bucket_size);
            int y_bucket = (int)(y_center / y_bucket_size);

            // calculate spatial buckets
            l_bucket = std::max(0, std::min(l_bucket, dims[0] - 1));
            a_bucket = std::max(0, std::min(a_bucket, dims[1] - 1));
            b_bucket = std::max(0, std::min(b_bucket, dims[2] - 1));
            x_bucket = std::max(0, std::min(x_bucket, dims[3] - 1));
            y_bucket = std::max(0, std::min(y_bucket, dims[4] - 1));

            // calculate hash key
            int hash_key = l_bucket;
            hash_key = hash_key * dims[1] + a_bucket;
            hash_key = hash_key * dims[2] + b_bucket;
            hash_key = hash_key * dims[3] + x_bucket;
            hash_key = hash_key * dims[4] + y_bucket;

            return hash_key;
        }

        // called for hashing segmented images, and storing them in the instance of this class
        // expects a cielab image for input_image
        void Hash(const cv::Mat& input_image, const cv::Mat& labels, int superpixel_count, unsigned long* pixel_count) {
            HashKey *superpixels = (HashKey*) calloc(superpixel_count, sizeof(HashKey));
            for (int row = 0; row < labels.rows; row++) {
                for (int col = 0; col < labels.cols; col++) {
                    cv::Vec3b lab_pixel = input_image.at<cv::Vec3b>(row, col);
                    int sp = labels.at<int>(row, col);
                    HashKey &curr = superpixels[sp];

                    // add to total color values to aid in calculating average color later
                    curr.l_tot += lab_pixel[0];
                    curr.a_tot += lab_pixel[1];
                    curr.b_tot += lab_pixel[2];

                    // set initial params indicating spatial extent
                    if (curr.pixel_count == 0) {
                        curr.x_range.first = col;
                        curr.x_range.second = col;
                        curr.y_range.first = row;
                        curr.y_range.second = row;
                        curr.original_image = &input_image;

                    // update spatial extent of superpixel if broader sections are discovered
                    } else {
                        if (curr.x_range.first > col) curr.x_range.first = col;
                        if (curr.x_range.second < col) curr.x_range.second = col;
                        if (curr.y_range.first > row) curr.y_range.first = row;
                        if (curr.y_range.second < row) curr.y_range.second = row;
                    }
                    curr.pixel_count += 1;
                    // hash superpixel if all subpixels have been found
                    if (curr.pixel_count == pixel_count[sp]) {
                        int key = calculate_hash_key(curr);
                        if (key != -1) {
                            hashTable[key].push_back(curr);
                        }
                    }
                }
            }
            free(superpixels);
        }
};


#endif