# Third homework assignment

## Multilayer Perceptron (MLP) and Digit Recognition

A Multilayer Perceptron (MLP) is a type of feedforward artificial neural network commonly used in supervised learning tasks, such as classification and regression. It consists of multiple layers of interconnected neurons, typically organized into an input layer, one or more hidden layers, and an output layer. Each neuron computes a weighted sum of its inputs, adds a bias term, and then passes this result through a non-linear activation function, like the sigmoid or ReLU function. This structure allows the MLP to model complex, non-linear relationships between input data and target outputs, making it particularly effective for problems like digit recognition.

Digit recognition is a classic machine learning task aimed at identifying handwritten digits from images, and it serves as an essential example of pattern recognition and classification problems. The (MNIST dataset)[https://www.kaggle.com/datasets/hojjatk/mnist-dataset] is the most widely used benchmark for digit recognition, consisting of 70,000 handwritten digit images labeled from 0 to 9. Each image is standardized to a size of 28×28 pixels, represented by grayscale values ranging from 0 to 255. Due to its simplicity and accessibility, MNIST is often employed as an introductory dataset for neural network experiments, allowing students and researchers to explore fundamental concepts in machine learning and neural network architectures. 

### Homework assignment 

In this assignment, you will be tasked with implementing the vectorization of inference for a Multilayer Perceptron (MLP) neural network, structured with an input layer, one hidden layer, and an output layer. The trained model parameters required for inference, such as weights and biases, are already provided in the weights folder. Your vectorization efforts should primarily focus on the `test` function, which performs the forward propagation step to classify input data based on the given model. Specifically, vectorization should target computations involving the input-to-hidden and hidden-to-output layer transformations and their associated activation functions. Your objective is to implement efficient vectorized computations using AVX2 instructions to significantly enhance inference performance.

### Vectorize program 

In the repository, you will find the scalar implementation of the MLP inference function, which is located in the `src` folder. Your task is to vectorize this function using AVX2 instructions. There are two files in the `src` folder: `main_test_double_sca` and `main_test_float_sca`. The `main_test_double_sca` file contains the scalar implementation of the MLP inference function using double precision floating point numbers, while the `main_test_float_sca` file contains the scalar implementation using single precision floating point numbers. Your goal is to vectorize both implementations.

### Evaluating the speedup from uttilizing AVX SIMD instructions 

Using the `time()` function, your task is to assess the execution time of the MLP inference function and compute the speedup achieved by the vectorized implementation compared to the scalar version. You are required to report the speedup ratio for the following scenarios:

1. Input data weights and biases represented as single precision floating point numbers. (5 points)
2. Input data, weights and biases represented as double precision floating point numbers. (5 points)

For each scenario, you will assess the performance for different sizes of hidden layers, specifically 128, 256, 512, and 1024 neurons. The other parameters of the MLP network, such as the number of input neurons and output neurons, will remain as defined in the provided code. Perform the measurements over five attempts and calculate the mean of the execution times. Use these results to determine the speedup ratio achieved by vectorization.


### Bonus challenge: Unleash the power of AVX512

As a bonus challenge, you are invited to further optimize the inference phase of the MLP network by leveraging AVX-512 instructions, available on the AMD Genoa processor series deployed on the cluster. This task involves extending your existing AVX2 vectorized solution to utilize the wider AVX-512 registers and instruction set, allowing for greater parallelism and potentially higher throughput. Focus on adapting the same test function, now optimized for AVX-512, and ensure that the implementation maintains the same numerical correctness as previous versions. Once completed, analyze the performance by comparing the runtime of scalar, AVX2, and AVX-512 implementations on the same input data. 


### Literature and additional materials

1. [Intel intristic guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL) 

### Analysis and Reporting:
Describe your results in a report (two-page max).


