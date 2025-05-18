#!/bin/bash
#SBATCH --job-name=mlp_benchmark
#SBATCH --output=mlp_benchmark.out
#SBATCH --error=mlp_benchmark.err
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# Compile all implementations
gcc -mavx2 -o main_test_double_sca main_test_double_sca.c -lm
gcc -mavx2 -o main_test_float_sca main_test_float_sca.c -lm
gcc -mavx2 -o main_test_double_vec main_test_double_vec.c -lm
gcc -mavx2 -o main_test_float_vec main_test_float_vec.c -lm

# Hidden layer sizes to test
hidden_sizes=(128 256 512 1024)

# Create results directory if it doesn't exist
mkdir -p ./results

# Function to add floating point numbers using awk
add_float() {
    echo "$1 $2" | awk '{print $1 + $2}'
}

# Function to divide floating point numbers using awk
div_float() {
    echo "$1 $2" | awk '{printf "%.10f", $1 / $2}'
}

# Run benchmark
for size in "${hidden_sizes[@]}"
do
    echo "Running benchmark with hidden layer size: $size"
    
    # Double precision scalar
    echo "Double precision scalar:"
    total_time_double_sca=0
    for i in {1..5}
    do
        time_output=$(srun ./main_test_double_sca $size)
        time=$(echo "$time_output" | grep "Average time" | awk '{print $3}')
        total_time_double_sca=$(add_float $total_time_double_sca $time)
        echo "  Run $i: $time seconds"
    done
    avg_time_double_sca=$(div_float $total_time_double_sca 5)
    echo "  Average time: $avg_time_double_sca seconds"
    
    # Double precision vectorized
    echo "Double precision vectorized:"
    total_time_double_vec=0
    for i in {1..5}
    do
        time_output=$(srun ./main_test_double_vec $size)
        time=$(echo "$time_output" | grep "Average time" | awk '{print $3}')
        total_time_double_vec=$(add_float $total_time_double_vec $time)
        echo "  Run $i: $time seconds"
    done
    avg_time_double_vec=$(div_float $total_time_double_vec 5)
    echo "  Average time: $avg_time_double_vec seconds"
    
    # Calculate speedup for double precision
    speedup_double=$(div_float $avg_time_double_sca $avg_time_double_vec)
    echo "  Speedup (double precision): $speedup_double"
    
    # Float precision scalar
    echo "Float precision scalar:"
    total_time_float_sca=0
    for i in {1..5}
    do
        time_output=$(srun ./main_test_float_sca $size)
        time=$(echo "$time_output" | grep "Average time" | awk '{print $3}')
        total_time_float_sca=$(add_float $total_time_float_sca $time)
        echo "  Run $i: $time seconds"
    done
    avg_time_float_sca=$(div_float $total_time_float_sca 5)
    echo "  Average time: $avg_time_float_sca seconds"
    
    # Float precision vectorized
    echo "Float precision vectorized:"
    total_time_float_vec=0
    for i in {1..5}
    do
        time_output=$(srun ./main_test_float_vec $size)
        time=$(echo "$time_output" | grep "Average time" | awk '{print $3}')
        total_time_float_vec=$(add_float $total_time_float_vec $time)
        echo "  Run $i: $time seconds"
    done
    avg_time_float_vec=$(div_float $total_time_float_vec 5)
    echo "  Average time: $avg_time_float_vec seconds"
    
    # Calculate speedup for float precision
    speedup_float=$(div_float $avg_time_float_sca $avg_time_float_vec)
    echo "  Speedup (float precision): $speedup_float"
    
    # Save results to file
    echo "Hidden Layer Size: $size" >> ./results/task1_results.txt
    echo "Double Precision - Scalar: $avg_time_double_sca seconds" >> ./results/task1_results.txt
    echo "Double Precision - Vectorized: $avg_time_double_vec seconds" >> ./results/task1_results.txt
    echo "Double Precision - Speedup: $speedup_double" >> ./results/task1_results.txt
    echo "Float Precision - Scalar: $avg_time_float_sca seconds" >> ./results/task1_results.txt
    echo "Float Precision - Vectorized: $avg_time_float_vec seconds" >> ./results/task1_results.txt
    echo "Float Precision - Speedup: $speedup_float" >> ./results/task1_results.txt
    echo "----------------------------------------" >> ./results/task1_results.txt
done

echo "Benchmark complete. Results saved to ./results/task1_results.txt" 