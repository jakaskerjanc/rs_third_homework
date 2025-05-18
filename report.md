# Task 1: Multilayer Perceptron (MLP) and Digit Recognition

## Introduction

This report presents the implementation and performance analysis of a Multilayer Perceptron (MLP) neural network for digit recognition using the MNIST dataset. The MLP consists of an input layer (784 neurons for 28×28 pixel images), a single hidden layer of variable size, and an output layer (10 neurons representing digits 0-9). The focus of this task was to vectorize the inference function using AVX2 SIMD instructions to achieve performance improvements over the scalar implementation.

## Implementation Approach

The vectorization was primarily focused on the `test` function, which performs forward propagation through the network. The key components of the vectorized implementation include:

### Float Precision Implementation
- Processes 8 neurons simultaneously using 256-bit AVX2 registers
- The input-to-hidden layer calculation was vectorized by:
  - Loading bias values using `_mm256_loadu_ps`
  - Broadcasting input values with `_mm256_set1_ps`
  - Setting weight vectors with `_mm256_set_ps`
  - Performing parallel multiplication and addition with `_mm256_mul_ps` and `_mm256_add_ps`

### Double Precision Implementation
- Processes 4 neurons simultaneously due to double-precision values taking twice the space
- Uses similar AVX2 instructions as the float implementation but with double-precision variants (`_mm256_loadu_pd`, `_mm256_set1_pd`, etc.)

### Vectorized Sigmoid Function
- Implemented specialized `sigmoid_avx2` functions for both float and double precision
- Calculates sigmoid activation for multiple values simultaneously
- Uses a combination of vector operations to compute 1/(1+exp(-x))

The implementation handles edge cases, such as when the number of neurons isn't a multiple of the vector width (8 for float, 4 for double), by processing the remainder with scalar operations.

## Performance Results

Testing was conducted with different hidden layer sizes (128, 256, 512, and 1024 neurons) for both single and double precision floating-point operations. The results show consistent speedup across all configurations:

| Hidden Layer Size | Double Precision Speedup | Float Precision Speedup |
|-------------------|--------------------------|-------------------------|
| 128               | 1.40x                    | 1.71x                   |
| 256               | 1.44x                    | 1.86x                   |
| 512               | 1.22x                    | 1.78x                   |
| 1024              | 2.38x                    | 1.96x                   |

### Key Observations:
- Float precision consistently achieves higher speedups (1.71x-1.96x) compared to double precision (1.22x-2.38x)
- The most significant speedup for double precision was observed with the largest hidden layer size (1024 neurons)
- Float precision shows more consistent performance across different layer sizes
- Both implementations show improved performance with larger layer sizes, particularly at 1024 neurons

## Why Vectorization is Faster

The vectorized implementation demonstrates improved performance for several reasons:

1. **Parallel Data Processing**: AVX2 instructions process multiple data elements simultaneously (8 floats or 4 doubles), reducing the total number of operations required.

2. **Reduced Loop Overhead**: Fewer iterations are needed for the main computation loops, decreasing branch prediction misses and loop control overhead.

3. **Efficient Memory Access**: Vectorized code can often make better use of cache lines by operating on contiguous memory blocks, reducing cache misses.

4. **Optimized Instruction Pipeline**: Modern CPUs are designed to efficiently execute SIMD instructions, with dedicated execution units that can perform vector operations with minimal latency.

5. **Reduced Function Call Overhead**: The vectorized sigmoid function processes multiple values at once, reducing the total number of function calls.

## AVX SIMD Instructions

Advanced Vector Extensions (AVX) is an x86 instruction set extension that provides enhanced support for Single Instruction, Multiple Data (SIMD) operations. Key aspects of AVX used in this implementation include:

### AVX2 Features
- Introduced in Intel Haswell architecture (2013)
- 256-bit wide registers (YMM0-YMM15) that can hold:
  - 8 single-precision floating-point values (32-bit floats)
  - 4 double-precision floating-point values (64-bit doubles)
- Rich set of instructions for arithmetic, logical, and data movement operations

### Key Instructions Used
- `_mm256_loadu_ps/pd`: Load unaligned packed values
- `_mm256_set1_ps/pd`: Broadcast a single value to all elements
- `_mm256_set_ps/pd`: Set vector elements individually
- `_mm256_add_ps/pd`: Add packed values
- `_mm256_mul_ps/pd`: Multiply packed values
- `_mm256_storeu_ps/pd`: Store unaligned packed values

### Theoretical vs. Achieved Speedup
- The theoretical maximum speedup would be 8x for float and 4x for double operations
- Our implementation achieved up to 1.96x for float and 2.38x for double
- The achieved speedups are lower than theoretical maximums due to:
  - Memory access patterns and cache behavior
  - Overhead from non-vectorized portions of the code
  - Memory allocation and initialization costs
  - Branch prediction and instruction pipeline effects

## Conclusion

The vectorization of the MLP inference using AVX2 SIMD instructions resulted in consistent performance improvements across all tested configurations. While the speedups are more modest than initially expected, they demonstrate the effectiveness of SIMD vectorization in neural network inference.

The float precision implementation shows more consistent performance across different layer sizes, while double precision shows more variation but achieves its best performance with larger layer sizes. This suggests that the choice between float and double precision should consider both the required numerical precision and the specific layer sizes being used.

The results indicate that while SIMD vectorization provides performance benefits, there is still room for optimization. Future work could explore:
1. Further optimization of memory access patterns
2. Investigation of cache behavior and potential improvements
3. Analysis of instruction pipeline utilization
4. Exploration of alternative vectorization strategies for different layer sizes

The vectorization of the MLP inference using AVX2 SIMD instructions resulted in substantial performance improvements, particularly for single-precision floating-point operations. The implementation successfully leveraged parallel processing capabilities to reduce computation time, with the most significant gains observed in the float precision implementation.

The float precision implementation consistently outperformed double precision in terms of speedup, which aligns with expectations given that AVX2 can process twice as many float values as double values simultaneously. This suggests that for neural network inference where precision requirements allow, using single-precision floating-point can provide significant performance benefits when combined with SIMD vectorization.

# Task 2: K-means Clustering with SIMD Vectorization

## Introduction

This section presents the implementation and performance analysis of the K-means clustering algorithm with SIMD vectorization using AVX2 instructions. K-means is an unsupervised learning algorithm that partitions data points into K clusters, where each data point belongs to the cluster with the nearest mean.

## Implementation Approach

The vectorization of the K-means algorithm focused on accelerating the distance calculation between data points and cluster centroids, which is the most computationally intensive part of the algorithm. Similar to Task 1, both float and double precision implementations were developed.

### Key Components of Vectorized Implementation:
- Vectorized Euclidean distance calculations between points and centroids
- Implemented parallel processing of multiple dimensions simultaneously
- Utilized AVX2 instructions for both float and double precision operations
- Optimized memory access patterns for improved cache utilization

## Performance Results

Testing was conducted with different numbers of clusters (4, 8, and 16) for both single and double precision floating-point operations. The results show:

| Precision | Clusters | SCA Cycles | VEC Cycles | Speedup (SCA/VEC) |
|-----------|----------|------------|------------|-------------------|
| double | 4 | 754511733 | 782808773 | 0.96 |
| double | 8 | 762217240 | 785181718 | 0.97 |
| double | 16 | 756383474 | 789883332 | 0.96 |
| float | 4 | 703422053 | 627284295 | 1.12 |
| float | 8 | 708101626 | 615494235 | 1.15 |
| float | 16 | 716112858 | 626315844 | 1.14 |

### Key Observations:
- Float precision implementation shows consistent speedup (1.12x-1.15x) across different cluster sizes
- Double precision implementation unexpectedly shows slight slowdown (0.96x-0.97x) compared to scalar implementation
- The number of clusters has minimal impact on the relative performance of vectorized vs. scalar implementations
- The best performance was achieved with float precision and 8 clusters (1.15x speedup)

## Analysis of Results

### Float Precision Success
The float precision implementation demonstrates successful vectorization with consistent speedups around 1.12x-1.15x. This is because:
- AVX2 processes 8 float values simultaneously, allowing efficient parallel distance calculations
- The memory access patterns in float precision align well with cache line sizes
- Fewer memory loads are required due to the smaller size of float values

### Double Precision Challenges
The double precision implementation shows slightly worse performance than scalar code (0.96x-0.97x). Potential reasons include:
- Memory access overhead might outweigh computational benefits
- Cache utilization may be less efficient with double precision values
- Vectorization overhead (loading/storing vectors) could exceed the benefits for the specific data patterns
- Only 4 doubles can be processed simultaneously with AVX2, reducing the potential parallelism

## Conclusion

The vectorization of K-means clustering using AVX2 SIMD instructions produced mixed results. While the float precision implementation showed consistent speedups, the double precision implementation slightly underperformed compared to scalar code.

These results highlight important considerations when applying SIMD vectorization:
1. The choice of precision significantly impacts vectorization benefits
2. Memory access patterns and cache behavior can be as important as computational parallelism
3. Smaller data types (float vs. double) often benefit more from vectorization due to higher parallelism
4. The nature of the algorithm and its memory access patterns may limit potential speedups

For K-means clustering with the tested dataset, float precision with SIMD vectorization provides the best performance, offering up to 1.15x speedup over scalar implementation. This suggests that where precision requirements allow, single-precision floating-point operations should be preferred when using SIMD vectorization for K-means clustering. 