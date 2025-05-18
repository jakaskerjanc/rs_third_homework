#!/bin/bash
#SBATCH --job-name=k-means-benchmark
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# Create results directory if it doesn't exist
mkdir -p results

# File to store results
RESULTS_FILE="results/task2_results.txt"

# Clean previous results
> $RESULTS_FILE

# Define versions, precisions, and cluster counts
VERSIONS=("sca" "vec")
PRECISIONS=("double" "float")
CLUSTER_COUNTS=(4 8 16)

# Function to compile all executables
compile_all() {
    echo "Compiling all versions..."
    
    for VERSION in "${VERSIONS[@]}"; do
        for PRECISION in "${PRECISIONS[@]}"; do
            for CLUSTERS in "${CLUSTER_COUNTS[@]}"; do
                SOURCE_FILE="k-means_${VERSION}_${PRECISION}.c"
                EXECUTABLE="k-means_${VERSION}_${PRECISION}_${CLUSTERS}"
                
                echo "Compiling $SOURCE_FILE with $CLUSTERS clusters..."
                gcc -mavx2 -DNUM_CLUSTERS=$CLUSTERS -o $EXECUTABLE $SOURCE_FILE -lm
                
                if [ $? -ne 0 ]; then
                    echo "Error compiling $SOURCE_FILE with $CLUSTERS clusters"
                    exit 1
                fi
            done
        done
    done
    
    echo "All versions compiled successfully"
    echo ""
}

# Function to run benchmark and collect results
run_benchmark() {
    VERSION=$1
    PRECISION=$2
    NUM_CLUSTERS=$3
    
    # Output filename
    EXECUTABLE="k-means_${VERSION}_${PRECISION}_${NUM_CLUSTERS}"
    
    echo "Running $EXECUTABLE..."
    echo "Testing $VERSION implementation with $PRECISION precision and $NUM_CLUSTERS clusters" >> $RESULTS_FILE
    
    # Use srun for HPC environment
    ./$EXECUTABLE >> $RESULTS_FILE
    
    # Get cycles from output
    cycles=$(grep "Time for K-means" $RESULTS_FILE | tail -n 1 | awk '{print $5}')
    echo "Cycles: $cycles" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
    
    # Return the cycles
    echo $cycles
}

# First compile all versions
compile_all

# Then run the benchmarks
for VERSION in "${VERSIONS[@]}"; do
    for PRECISION in "${PRECISIONS[@]}"; do
        for CLUSTERS in "${CLUSTER_COUNTS[@]}"; do
            run_benchmark $VERSION $PRECISION $CLUSTERS
        done
    done
done