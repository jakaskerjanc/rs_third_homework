#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <immintrin.h> // For AVX2 intrinsics

#define INPUT_NODES 784  // 28*28 pixels
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000

float training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
float training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
float test_images[NUM_TEST_IMAGES][INPUT_NODES];
float test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Dynamically allocated weights and biases
float **weight1;
float **weight2;
float *bias1;
float *bias2;

int HIDDEN_NODES;
int NUMBER_OF_EPOCHS;

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Vectorized sigmoid for 8 floats at once using AVX2
__m256 sigmoid_avx2(__m256 x)
{
    // Constants for sigmoid approximation
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zeros = _mm256_setzero_ps();
    
    // Calculate -x
    __m256 neg_x = _mm256_sub_ps(zeros, x);
    
    // In-place implementation of exp(-x)
    __m256 exp_neg_x = _mm256_setzero_ps();
    
    // Use scalar exp for each element
    float temp[8];
    _mm256_storeu_ps(temp, neg_x);
    
    temp[0] = expf(temp[0]);
    temp[1] = expf(temp[1]);
    temp[2] = expf(temp[2]);
    temp[3] = expf(temp[3]);
    temp[4] = expf(temp[4]);
    temp[5] = expf(temp[5]);
    temp[6] = expf(temp[6]);
    temp[7] = expf(temp[7]);
    
    exp_neg_x = _mm256_loadu_ps(temp);
    
    // Calculate 1 + exp(-x)
    __m256 denominator = _mm256_add_ps(ones, exp_neg_x);
    
    // Calculate 1 / (1 + exp(-x))
    __m256 result = _mm256_div_ps(ones, denominator);
    
    return result;
}

int max_index(float arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

void load_mnist()
{
    // Open the training images file
    FILE *training_images_file = fopen("./mnist_dataset/mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    // Open the training labels file
    FILE *training_labels_file = fopen("./mnist_dataset/mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    // Open the test images file
    FILE *test_images_file = fopen("./mnist_dataset/mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    // Open the test labels file
    FILE *test_labels_file = fopen("./mnist_dataset/mnist_test_labels.bin", "rb");
    if (test_labels_file == NULL)
    {
        printf("Error opening test labels file\n");
        exit(1);
    }

    // Read the training images
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (float)pixel / 255.0f;
        }
    }

    // Read the training labels
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                training_labels[i][j] = 1.0f;
            }
            else
            {
                training_labels[i][j] = 0.0f;
            }
        }
    }

    // Read the test images
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (float)pixel / 255.0f;
        }
    }

    // Read the test labels
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                test_labels[i][j] = 1.0f;
            }
            else
            {
                test_labels[i][j] = 0.0f;
            }
        }
    }

    // Close the files
    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

int test(float input[INPUT_NODES], float** weight1, float** weight2, float* bias1, float* bias2, int correct_label)
{   
    int correct_predictions = 0;
    float hidden[HIDDEN_NODES];
    float output_layer[OUTPUT_NODES];
    
    // Vectorized feedforward from input to hidden layer
    int i, j;
    const int vec_size = 8; // AVX2 processes 8 floats at once
    
    // Process hidden neurons in chunks of 8 using AVX2
    for (i = 0; i < HIDDEN_NODES; i += vec_size) {
        int remaining = HIDDEN_NODES - i;
        int current_chunk = remaining < vec_size ? remaining : vec_size;
        
        if (current_chunk == vec_size) {
            // Process a full vector of 8 neurons
            __m256 sum_vec = _mm256_loadu_ps(&bias1[i]);
            
            // Accumulate dot products for each input
            for (j = 0; j < INPUT_NODES; j++) {
                __m256 input_val = _mm256_set1_ps(input[j]);
                __m256 weight_vec = _mm256_set_ps(
                    weight1[j][i+7],
                    weight1[j][i+6],
                    weight1[j][i+5],
                    weight1[j][i+4],
                    weight1[j][i+3],
                    weight1[j][i+2],
                    weight1[j][i+1],
                    weight1[j][i]
                );
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(input_val, weight_vec));
            }
            
            // Apply sigmoid function to the accumulated values
            __m256 result = sigmoid_avx2(sum_vec);
            
            // Store the results
            _mm256_storeu_ps(&hidden[i], result);
        } else {
            // Handle the remaining neurons using scalar operations
            for (int k = 0; k < current_chunk; k++) {
                float sum = bias1[i + k];
                for (j = 0; j < INPUT_NODES; j++) {
                    sum += input[j] * weight1[j][i + k];
                }
                hidden[i + k] = sigmoid(sum);
            }
        }
    }
    
    // Vectorized feedforward from hidden to output layer
    if (OUTPUT_NODES >= vec_size) {
        for (i = 0; i < OUTPUT_NODES; i += vec_size) {
            int remaining = OUTPUT_NODES - i;
            int current_chunk = remaining < vec_size ? remaining : vec_size;
            
            if (current_chunk == vec_size) {
                // Process a full vector of 8 neurons
                __m256 sum_vec = _mm256_loadu_ps(&bias2[i]);
                
                // Accumulate dot products for each hidden neuron
                for (j = 0; j < HIDDEN_NODES; j++) {
                    __m256 hidden_val = _mm256_set1_ps(hidden[j]);
                    __m256 weight_vec = _mm256_set_ps(
                        weight2[j][i+7],
                        weight2[j][i+6],
                        weight2[j][i+5],
                        weight2[j][i+4],
                        weight2[j][i+3],
                        weight2[j][i+2],
                        weight2[j][i+1],
                        weight2[j][i]
                    );
                    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(hidden_val, weight_vec));
                }
                
                // Apply sigmoid function to the accumulated values
                __m256 result = sigmoid_avx2(sum_vec);
                
                // Store the results
                _mm256_storeu_ps(&output_layer[i], result);
            } else {
                // Handle the remaining neurons using scalar operations
                for (int k = 0; k < current_chunk; k++) {
                    float sum = bias2[i + k];
                    for (j = 0; j < HIDDEN_NODES; j++) {
                        sum += hidden[j] * weight2[j][i + k];
                    }
                    output_layer[i + k] = sigmoid(sum);
                }
            }
        }
    } else {
        // If output nodes < 8, use scalar approach
        for (i = 0; i < OUTPUT_NODES; i++) {
            float sum = bias2[i];
            for (j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid(sum);
        }
    }
    
    int index = max_index(output_layer, OUTPUT_NODES);

    correct_predictions = index == correct_label ? 1 : 0;
     
    return correct_predictions;
}

// utils
void allocate_memory() {
    weight1 = malloc(INPUT_NODES * sizeof(float *));
    for (int i = 0; i < INPUT_NODES; i++) {
        weight1[i] = malloc(HIDDEN_NODES * sizeof(float));
    }

    weight2 = malloc(HIDDEN_NODES * sizeof(float *));
    for (int i = 0; i < HIDDEN_NODES; i++) {
        weight2[i] = malloc(OUTPUT_NODES * sizeof(float));
    }

    bias1 = malloc(HIDDEN_NODES * sizeof(float));
    bias2 = malloc(OUTPUT_NODES * sizeof(float));
}

void free_memory() {
    for (int i = 0; i < INPUT_NODES; i++) {
        free(weight1[i]);
    }
    free(weight1);

    for (int i = 0; i < HIDDEN_NODES; i++) {
        free(weight2[i]);
    }
    free(weight2);
    free(bias1);
    free(bias2);
}

void init_weigths_and_biases() {
    // Initialize weights and biases with random values
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weight1[i][j] = 0.1f; // Random value between -1 and 1
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weight2[i][j] = 0.1f; // Random value between -1 and 1
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        bias1[i] = 0.1f; // Random value between -1 and 1
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        bias2[i] = 0.1f; // Random value between -1 and 1
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <hidden_nodes>\n", argv[0]);
        return 1;
    }

    HIDDEN_NODES = atoi(argv[1]);

    // load weights and biases
    allocate_memory();
    init_weigths_and_biases();
    
    // load mnist dataset
    load_mnist();
    int correct_outcomes; 
    
    // measuring time
    clock_t start, end;
    double cpu_time_used = 0.0;

    cpu_time_used = 0.0;
    // Train the network
    correct_outcomes = 0;
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        start = clock();
        correct_outcomes += test(training_images[i], weight1, weight2, bias1, bias2, max_index(training_labels[i], OUTPUT_NODES));
        end = clock();
        cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    printf("Average time: %f seconds\n", cpu_time_used / NUM_TRAINING_IMAGES);

    free_memory();
    return 0;
} 