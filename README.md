# Third homework assignment

## K-means clustering and image processing 

The k-means algorithm is an unsupervised machine-learning technique used for clustering data into distinct groups based on similarities in their features. In image processing, K-means finds applications in tasks like image segmentation and color quantization. K-means can partition an image into segments with similar pixel values by treating each pixel as a data point in a high-dimensional space. This segmentation is beneficial in various image analysis tasks, such as object detection, image compression, and image enhancement. For instance, for color quantization, K-means clusters similar colors together, reducing the number of distinct colors in an image while preserving its visual quality. This makes the image easier to process and store without significant loss of information. The K-means algorithm offers an efficient and practical approach to analyzing and manipulating images in various applications.

### Homework assignment 

In this assignment, you will be tasked with implementing the vectorization of the K-means algorithm for grayscale-level quantization. The test image provided for evaluation is named "bosko_k-means.jpg," upon execution of the program, it will output an image "bosko_k-means.jpg" after processing. Within the codebase of the scalar version, two critical functions require vectorization. Firstly, the "assign_points_to_clusters" function assigns pixels to their nearest cluster based on the difference between each pixel and the cluster centroid. Secondly, the "update_centroids" function recalculates the centroids of clusters based on the pixels assigned to each cluster. The K-means algorithm iteratively applies these two functions until the change in centroid values falls below a specified threshold. Your task is to implement efficient vectorized versions of these functions to enhance the performance of the K-means algorithm for grayscale-level quantization.

### Vectorize program 

In the repository, you will find the scalar implementation of the K-means algorithm (k-means_sca.c). Using scalar implementation, explicitly vectorize the program using AVX/AV2 intrinsics. 

### Evaluating the speedup from uttilizing AVX SIMD instructions 

Using the rdtsc() function, your task is to assess the execution time of the K-means algorithm and compute the speedup achieved by the vectorized implementation compared to the scalar version. You are required to report the speedup ratio for the following scenarios:
1. Utilizing 4 clusters with pixel data represented as 8-bit unsigned numbers.
2. Utilizing 8 clusters with pixel data represented as 8-bit unsigned numbers.
3. Utilizing 16 clusters with pixel data represented as 8-bit unsigned numbers.
4. Utilizing 4 clusters with pixel data represented as single precision floating point numbers.
5. Utilizing 8 clusters with pixel data represented as single precision floating point numbers.
6. Utilizing 16 clusters with pixel data represented as single precision floating point numbers.
7. Utilizing 4 clusters with pixel data represented as double precision floating point numbers.
8. Utilizing 8 clusters with pixel data represented as double precision floating point numbers.
9. Utilizing 16 clusters with pixel data represented as double precision floating point numbers.

For each scenario, measure the execution time of both the scalar and vectorized versions of the K-means algorithm using rdtsc(), then calculate the ratio of execution times to determine the speedup achieved by vectorization. 

### Bonus challenge: SIMD competition 

To add a little bit of excitement, we will host a mini competition where you will be required to report the cycle counts of execution, averaged over ten attempts. You will submit your results' mean and standard deviation on the e-classroom platform. Additionally, the top five students with the shortest execution times will be awarded an additional 5 points. However, please note that we will thoroughly check your execution times, and any evidence of cheating will result in a deduction of 5 points.

### Literature and additional materials

1. [Intel intristic guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL) 

### Analysis and Reporting:
Describe your results in a report (two-page max).


