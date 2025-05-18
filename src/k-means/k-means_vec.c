#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>

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

struct Cluster {
    double centroid;
    int num_points;
    int* points;
};

// Vectorized function to assign points to clusters
void assign_points_to_clusters(struct Cluster clusters[], double* image, int image_size) {
    __m256d min_distances;
    __m256d distances;
    __m256d pixel_vec;
    __m256d centroid_vec;
    double min_distance;
    int cluster_index;
    double* d = (double*)aligned_alloc(32, sizeof(double) * NUM_CLUSTERS);

    for (int i = 0; i < image_size; i++) {
        pixel_vec = _mm256_set1_pd(image[i]);
        
        // Process clusters in groups of 4 (AVX2 can handle 4 doubles at once)
        for (int k = 0; k < NUM_CLUSTERS; k += 4) {
            centroid_vec = _mm256_loadu_pd(&clusters[k].centroid);
            distances = _mm256_sub_pd(pixel_vec, centroid_vec);
            distances = _mm256_abs_pd(distances);
            _mm256_storeu_pd(&d[k], distances);
        }

        // Find minimum distance and corresponding cluster
        min_distance = d[0];
        cluster_index = 0;
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            if (d[k] < min_distance) {
                min_distance = d[k];
                cluster_index = k;
            }
        }
        clusters[cluster_index].points[clusters[cluster_index].num_points++] = i;
    }

    free(d);
}

// Vectorized function to update centroids
void update_centroids(struct Cluster clusters[], double* image, int image_size) {
    __m256d sum_vec;
    __m256d pixel_vec;
    double sum;

    for (int i = 0; i < NUM_CLUSTERS; i++) {
        sum_vec = _mm256_setzero_pd();
        int remaining = clusters[i].num_points % 4;
        int aligned_size = clusters[i].num_points - remaining;

        // Process points in groups of 4
        for (int k = 0; k < aligned_size; k += 4) {
            pixel_vec = _mm256_set_pd(
                image[clusters[i].points[k + 3]],
                image[clusters[i].points[k + 2]],
                image[clusters[i].points[k + 1]],
                image[clusters[i].points[k]]
            );
            sum_vec = _mm256_add_pd(sum_vec, pixel_vec);
        }

        // Handle remaining points
        sum = 0.0;
        for (int k = aligned_size; k < clusters[i].num_points; k++) {
            sum += image[clusters[i].points[k]];
        }

        // Reduce sum_vec to scalar
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum_vec);
        sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum;

        clusters[i].centroid = sum / clusters[i].num_points;
    }
}

void k_means(double* image, int image_size, struct Cluster* clusters) {
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
        update_centroids(clusters, image, image_size);

        // Calculate difference between old and new centroids
        error = 0;
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            error += fabs(clusters[i].centroid - clusters_temp[i].centroid);
        }
        iterations++;

    } while (error > THRESHOLD && iterations < MAX_ITERATIONS);
}

void segment_image(double *image, struct Cluster* clusters, int image_size) {
    __m256d min_distances;
    __m256d distances;
    __m256d pixel_vec;
    __m256d centroid_vec;
    double min_distance;
    int cluster_index;
    double* d = (double*)aligned_alloc(32, sizeof(double) * NUM_CLUSTERS);

    for (int i = 0; i < image_size; i++) {
        pixel_vec = _mm256_set1_pd(image[i]);
        
        // Process clusters in groups of 4
        for (int k = 0; k < NUM_CLUSTERS; k += 4) {
            centroid_vec = _mm256_loadu_pd(&clusters[k].centroid);
            distances = _mm256_sub_pd(pixel_vec, centroid_vec);
            distances = _mm256_abs_pd(distances);
            _mm256_storeu_pd(&d[k], distances);
        }

        // Find minimum distance and corresponding cluster
        min_distance = d[0];
        cluster_index = 0;
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            if (d[k] < min_distance) {
                min_distance = d[k];
                cluster_index = k;
            }
        }
        image[i] = clusters[cluster_index].centroid * 255.0;
    }

    free(d);
}

int main() {
    long long unsigned int start, end, cycles;

    // Load image
    char image_name[] = "./bosko_grayscale.jpg";
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(image_name, &width, &height, &cpp, STBI_grey);
    if (h_imageIn == NULL) {
        printf("Error reading loading image %s!\n", image_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_name, width, height);
    printf("Image is %d bytes per pixel.\n", cpp);

    // Allocate and initialize image data
    double *image_pixels = (double*)aligned_alloc(32, sizeof(double) * width * height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_pixels[i*width + j] = h_imageIn[i * width + j]/255.0;
        }
    }

    int image_size = width * height;

    // Initialize clusters
    struct Cluster clusters[NUM_CLUSTERS];
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].points = (int*)malloc(sizeof(int) * image_size);
        clusters[i].centroid = i*1.0/(NUM_CLUSTERS-1);
        clusters[i].num_points = 0;
    }

    // Perform K-means clustering
    start = rdtsc();
    k_means(image_pixels, image_size, clusters);
    end = rdtsc();
    cycles = end - start;
    printf("Time for K-means: %lld cycles\n", cycles);

    // Print cluster centroids
    printf("Cluster centroids:\n");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Cluster %d: %.2f\n", i + 1, clusters[i].centroid);
    }

    // Segment image
    segment_image(image_pixels, clusters, image_size);

    // Save processed image
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_imageIn[i*width + j] = (char)(image_pixels[i*width + j]);
        }
    }
    stbi_write_jpg("bosko_k-means.jpg", width, height, STBI_grey, h_imageIn, 100);

    // Free memory
    free(image_pixels);
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        free(clusters[i].points);
    }
    stbi_image_free(h_imageIn);

    return 0;
} 