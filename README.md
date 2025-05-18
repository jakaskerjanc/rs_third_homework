# Third homework assignment

## Task 1: Multilayer Perceptron (MLP) and Digit Recognition

A Multilayer Perceptron (MLP) is a type of feedforward artificial neural network commonly used in supervised learning tasks, such as classification and regression. It consists of multiple layers of interconnected neurons, typically organized into an input layer, one or more hidden layers, and an output layer. Each neuron computes a weighted sum of its inputs, adds a bias term, and then passes this result through a non-linear activation function, like the sigmoid or ReLU function. This structure allows the MLP to model complex, non-linear relationships between input data and target outputs, making it particularly effective for problems like digit recognition.

Digit recognition is a classic machine learning task aimed at identifying handwritten digits from images, and it serves as an essential example of pattern recognition and classification problems. The (MNIST dataset)[https://www.kaggle.com/datasets/hojjatk/mnist-dataset] is the most widely used benchmark for digit recognition, consisting of 70,000 handwritten digit images labeled from 0 to 9. Each image is standardized to a size of 28×28 pixels, represented by grayscale values ranging from 0 to 255. Due to its simplicity and accessibility, MNIST is often employed as an introductory dataset for neural network experiments, allowing students and researchers to explore fundamental concepts in machine learning and neural network architectures. 

### Assignment 

In this assignment, you will be tasked with implementing the vectorization of inference for a Multilayer Perceptron (MLP) neural network, structured with an input layer, one hidden layer, and an output layer. The weights and biases are initialized to 0.1, as the focus of this homework is on execution speed rather than model accuracy. Your vectorization efforts should primarily focus on the `test` function, which performs the forward propagation step to classify input data based on the given model. Specifically, vectorization should target computations involving the input-to-hidden and hidden-to-output layer transformations and their associated activation functions. Your objective is to implement efficient vectorized computations using AVX2 instructions to significantly enhance inference performance.

### Vectorize program 

In the repository, you will find the scalar implementation of the MLP inference function, which is located in the `mlp` folder. Your task is to vectorize this function using AVX2 instructions. There are two files in the `mlp` folder: `main_test_double_sca` and `main_test_float_sca`. The `main_test_double_sca` file contains the scalar implementation of the MLP inference function using double precision floating point numbers, while the `main_test_float_sca` file contains the scalar implementation using single precision floating point numbers. Your goal is to vectorize both implementations.

### Evaluating the speedup from uttilizing AVX SIMD instructions 

Using the `time()` function, your task is to assess the execution time of the MLP inference function and compute the speedup achieved by the vectorized implementation compared to the scalar version. You are required to report the speedup ratio for the following scenarios:

1. Input data weights and biases represented as single precision floating point numbers. (3 points)
2. Input data, weights and biases represented as double precision floating point numbers. (2 points)

For each scenario, you will assess the performance for different sizes of hidden layers, specifically 128, 256, 512, and 1024 neurons. The other parameters of the MLP network, such as the number of input neurons and output neurons, will remain as defined in the provided code. Perform the measurements over five attempts and calculate the mean of the execution times. Use these results to determine the speedup ratio achieved by vectorization.

## Task 2: K-means clustering and image processing 

The k-means algorithm is an unsupervised machine-learning technique used for clustering data into distinct groups based on similarities in their features. In image processing, K-means finds applications in tasks like image segmentation and color quantization. K-means can partition an image into segments with similar pixel values by treating each pixel as a data point in a high-dimensional space. This segmentation is beneficial in various image analysis tasks, such as object detection, image compression, and image enhancement. For instance, for color quantization, K-means clusters similar colors together, reducing the number of distinct colors in an image while preserving its visual quality. This makes the image easier to process and store without significant loss of information. The K-means algorithm offers an efficient and practical approach to analyzing and manipulating images in various applications.

### Assignment 

In this assignment, you will be tasked with implementing the vectorization of the K-means algorithm for grayscale-level quantization. The test image provided for evaluation is named `bosko_grayscale.jpg,` upon execution of the program, it will output an image `bosko_k-means.jpg` after processing. Within the codebase of the scalar version, two critical functions require vectorization. Firstly, the `assign_points_to_clusters` function assigns pixels to their nearest cluster based on the difference between each pixel and the cluster centroid. Secondly, the `update_centroids` function recalculates the centroids of clusters based on the pixels assigned to each cluster. The K-means algorithm iteratively applies these two functions until the change in centroid values falls below a specified threshold. Your task is to implement efficient vectorized versions of these functions to enhance the performance of the K-means algorithm for grayscale-level quantization.

### Vectorize program 

In the folder `k-means`, you will find the scalar implementation of the K-means algorithm (k-means_sca.c). Using scalar implementation, explicitly vectorize the program using AVX/AV2 intrinsics. 

### Evaluating the speedup from uttilizing AVX SIMD instructions 

Using the `rdtsc()` function, your task is to assess the execution time of the K-means algorithm and compute the speedup achieved by the vectorized implementation compared to the scalar version. You are required to report the speedup ratio for the following scenarios:

1. Utilizing 4 clusters with pixel data represented as single precision floating point numbers. (1 point)
2. Utilizing 8 clusters with pixel data represented as single precision floating point numbers. (1 point)
3. Utilizing 4 clusters with pixel data represented as double precision floating point numbers. (1 point)
4. Utilizing 8 clusters with pixel data represented as double precision floating point numbers. (1 point)
5. Utilizing 16 clusters with pixel data represented as double precision floating point numbers. (1 point)

For each scenario, measure the execution time of both the scalar and vectorized versions of the K-means algorithm using `rdtsc()`, then calculate the ratio of execution times to determine the speedup achieved by vectorization. 

- Create 4 files, `k-means_sca_double.c, k-means_sca_float.c, k-means_vec_double.c, k-means_vec_float.c` 
- Number of clusters is passed as `NUM_CLUSTERS`

- Create `run_benchmark.sh`, where the program is benchmarked.
- Utilize `srun` (we are on HPC).
- Write the results to `results/task2_results.txt`.

## Literature and additional materials

1. [Intel intristic guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL) 
