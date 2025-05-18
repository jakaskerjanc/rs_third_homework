#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // AVX2 intrinsics

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NUM_CLUSTERS 4
#define MAX_ITERATIONS 10000
#define THRESHOLD 0.0001


unsigned long long rdtsc() {
    unsigned int hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// Define cluster structure
struct Cluster {
    double centroid;
    int num_points;
    int* points;
};

// Vectorized function to assign points (pixels) to clusters
void assign_points_to_clusters(struct Cluster clusters[], double* image, int image_size) {
    // Vector to store centroids
    double centroids[NUM_CLUSTERS];
    
    // Extract centroids into an array for easier vectorization
    for (int k = 0; k < NUM_CLUSTERS; k++) {
        centroids[k] = clusters[k].centroid;
    }
    
    // Reset point counts
    for (int k = 0; k < NUM_CLUSTERS; k++) {
        clusters[k].num_points = 0;
    }
    
    // Process 4 pixels at a time (AVX2 has 256-bit registers = 4 doubles)
    int i = 0;
    for (; i <= image_size - 4; i += 4) {
        // Load 4 pixels
        __m256d pixels = _mm256_loadu_pd(&image[i]);
        
        // Find closest centroid for each pixel
        int closest_cluster[4] = {0};
        double min_distances[4];
        
        // Initialize min_distances with distances to first centroid
        __m256d centroid_vec = _mm256_set1_pd(centroids[0]);
        __m256d diff = _mm256_sub_pd(pixels, centroid_vec);
        __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // Absolute value
        _mm256_storeu_pd(min_distances, abs_diff);
        
        // Check other centroids
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            centroid_vec = _mm256_set1_pd(centroids[k]);
            diff = _mm256_sub_pd(pixels, centroid_vec);
            abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // Absolute value
            
            // Store current distances
            double current_distances[4];
            _mm256_storeu_pd(current_distances, abs_diff);
            
            // Compare with minimum distances
            for (int j = 0; j < 4; j++) {
                if (current_distances[j] < min_distances[j]) {
                    min_distances[j] = current_distances[j];
                    closest_cluster[j] = k;
                }
            }
        }
        
        // Assign pixels to clusters
        for (int j = 0; j < 4; j++) {
            clusters[closest_cluster[j]].points[clusters[closest_cluster[j]].num_points++] = i + j;
        }
    }
    
    // Handle remaining pixels (fewer than 4)
    for (; i < image_size; i++) {
        int cluster_index = 0;
        double min_distance = fabs(image[i] - centroids[0]);
        
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            double d = fabs(image[i] - centroids[k]);
            if (d < min_distance) {
                min_distance = d;
                cluster_index = k;
            }
        }
        
        clusters[cluster_index].points[clusters[cluster_index].num_points++] = i;
    }
}

// Vectorized function to update centroids of clusters
void update_centroids(struct Cluster clusters[], double* image, int image_size) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (clusters[i].num_points == 0) {
            clusters[i].centroid = 0.0;
            continue;
        }
        
        __m256d sum_vector = _mm256_setzero_pd();
        int k = 0;
        
        // Process 4 points at a time
        for (; k <= clusters[i].num_points - 4; k += 4) {
            // Load 4 indices
            int indices[4];
            for (int j = 0; j < 4; j++) {
                indices[j] = clusters[i].points[k + j];
            }
            
            // Load 4 pixel values
            __m256d pixels = _mm256_set_pd(
                image[indices[3]], image[indices[2]], 
                image[indices[1]], image[indices[0]]
            );
            
            // Add to sum
            sum_vector = _mm256_add_pd(sum_vector, pixels);
        }
        
        // Sum the 4 doubles in the vector
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_vector);
        double sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        
        // Handle remaining points
        for (; k < clusters[i].num_points; k++) {
            int pixel_index = clusters[i].points[k];
            sum += image[pixel_index];
        }
        
        clusters[i].centroid = sum / clusters[i].num_points;
    }
}

// K-means clustering function
void k_means(double* image, int image_size, struct Cluster* clusters) {
    // Initialize clusters
    struct Cluster clusters_temp[NUM_CLUSTERS];
    double error = 0;
    int iterations = 0;

    do {
        // Save old clusters
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            clusters_temp[i] = clusters[i];
        }
        
        // Reinitialize cluster points
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            clusters[i].num_points = 0;
        }        

        assign_points_to_clusters(clusters, image, image_size);
        // Update centroids
        update_centroids(clusters, image, image_size);

        // Calculate difference between old and new centroids
        error = 0;
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            error += fabs(clusters[i].centroid - clusters_temp[i].centroid);
        }
        iterations++;

    } while (error > THRESHOLD && iterations < MAX_ITERATIONS);
}

// Function to segment image based on cluster values (vectorized)
void segment_image(double* image, struct Cluster* clusters, int image_size) {
    // Extract centroids to an array
    double centroids[NUM_CLUSTERS];
    double centroid_values[NUM_CLUSTERS];
    
    for (int k = 0; k < NUM_CLUSTERS; k++) {
        centroids[k] = clusters[k].centroid;
        centroid_values[k] = centroids[k] * 255.0;
    }

    int i = 0;
    // Process 4 pixels at a time
    for (; i <= image_size - 4; i += 4) {
        // Load 4 pixels
        __m256d pixels = _mm256_loadu_pd(&image[i]);
        
        // Find closest centroid for each pixel
        int closest_cluster[4] = {0};
        double min_distances[4];
        
        // Initialize min_distances with distances to first centroid
        __m256d centroid_vec = _mm256_set1_pd(centroids[0]);
        __m256d diff = _mm256_sub_pd(pixels, centroid_vec);
        __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // Absolute value
        _mm256_storeu_pd(min_distances, abs_diff);
        
        // Check other centroids
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            centroid_vec = _mm256_set1_pd(centroids[k]);
            diff = _mm256_sub_pd(pixels, centroid_vec);
            abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff); // Absolute value
            
            // Store current distances
            double current_distances[4];
            _mm256_storeu_pd(current_distances, abs_diff);
            
            // Compare with minimum distances
            for (int j = 0; j < 4; j++) {
                if (current_distances[j] < min_distances[j]) {
                    min_distances[j] = current_distances[j];
                    closest_cluster[j] = k;
                }
            }
        }
        
        // Set pixel values based on closest centroid
        double new_values[4];
        for (int j = 0; j < 4; j++) {
            new_values[j] = centroid_values[closest_cluster[j]];
        }
        
        __m256d new_pixels = _mm256_loadu_pd(new_values);
        _mm256_storeu_pd(&image[i], new_pixels);
    }
    
    // Handle remaining pixels
    for (; i < image_size; i++) {
        double min_distance = fabs(image[i] - centroids[0]);
        int cluster_index = 0;
        
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            double d = fabs(image[i] - centroids[k]);
            if (d < min_distance) {
                min_distance = d;
                cluster_index = k;
            }
        }
        
        image[i] = centroid_values[cluster_index];
    }
}

int main() {
    // Define sample grayscale image
    long long unsigned int start, end, cycles;

    // Load image from file and allocate space for the output image
    char image_name[] = "./bosko_grayscale.jpg";
    int width, height, cpp;
    // load only gray scale image
    unsigned char *h_imageIn = stbi_load(image_name, &width, &height, &cpp, STBI_grey);
    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", image_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_name, width, height);
    printf("Image is %d bytes per pixel.\n", cpp);
    
    double *image_pixels = (double*)malloc(sizeof(double) * width * height);
    // convert to grayscale 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_pixels[i*width + j] = h_imageIn[i * width + j]/255.0;
        }
    }

    int image_size = width * height;

    // cluster centroids
    struct Cluster clusters[NUM_CLUSTERS];
    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].points = (int*)malloc(sizeof(int) * image_size);
    }
    
    // Initialize centroids
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].centroid = i*1.0/(NUM_CLUSTERS-1);
        clusters[i].num_points = 0;
    }
    
    // Perform K-means clustering

    // warm up
    k_means(image_pixels, image_size, clusters);
    
    for (int iter = 1; iter <= 5; iter++) {
        printf("Iteration %d:\n", iter);
        start = rdtsc();
        k_means(image_pixels, image_size, clusters);
        end = rdtsc();
        cycles = end - start;
        printf("Time for K-means: %lld cycles\n", cycles);
    }

    //print cluster centroids
    printf("Cluster centroids:\n");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Cluster %d: %.2f\n", i + 1, clusters[i].centroid);
    }

    // Segment image
    segment_image(image_pixels, clusters, image_size);

    // Save image to file
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_imageIn[i*width + j] = (unsigned char)(image_pixels[i*width + j]);
        }
    }

    stbi_write_jpg("bosko_k-means.jpg", width, height, STBI_grey, h_imageIn, 100);
    
    // Free memory
    free(image_pixels);
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        free(clusters[i].points);
    }
    free(h_imageIn);

    return 0;
} 