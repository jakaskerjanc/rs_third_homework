#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>


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
    float centroid;
    int num_points;
    int* points;
};


// Function to assign points (pixels) to clusters
void assign_points_to_clusters(struct Cluster clusters[], float* image, int image_size) {
    int cluster_index = 0;
    float min_distance = 100000.0f;
    float* d = (float*)malloc(sizeof(float) * NUM_CLUSTERS);

    for (int i = 0; i < image_size; i++) { 
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            d[k] = fabsf(image[i] - clusters[k].centroid);
        }
        // argmin of array d 
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

// Function to update centroids of clusters
void update_centroids(struct Cluster clusters[], float* image, int image_size) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        float sum = 0;
        for (int k = 0; k < clusters[i].num_points; k++) {
            int pixel_index = clusters[i].points[k];
            sum += image[pixel_index];
        }
        clusters[i].centroid = sum / clusters[i].num_points;
    }
}

// K-means clustering function
void k_means(float* image, int image_size, struct Cluster* clusters) {
    
    // Initialize clusters
    struct Cluster clusters_temp[NUM_CLUSTERS];
    float error = 0;
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
            error += fabsf(clusters[i].centroid - clusters_temp[i].centroid);
        }
        iterations++;

    } while (error > THRESHOLD && iterations < MAX_ITERATIONS);
}

// Function to segment image based on cluster values
void segment_image(float *image, struct Cluster* clusters, int image_size) {
    for (int i = 0; i < image_size; i++) {
        float min_distance = fabsf(image[i] - clusters[0].centroid);
        int cluster_index = 0;
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            float d = fabsf(image[i] - clusters[k].centroid);
            if (d < min_distance) {
                min_distance = d;
                cluster_index = k;
            }
        }
        image[i] = (float)clusters[cluster_index].centroid*255.0f;
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
    
    float *image_pixels = (float*)malloc(sizeof(float) * width * height);
    // convert to grayscale 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_pixels[i*width + j] = h_imageIn[i * width + j]/255.0f;
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
        clusters[i].centroid = i*1.0f/(NUM_CLUSTERS-1);
        clusters[i].num_points = 0;
    }
    

    // warm up
    k_means(image_pixels, image_size, clusters);
    
    // Perform K-means clustering
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