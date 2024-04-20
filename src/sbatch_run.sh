#!/bin/bash

# Job name:
#SBATCH --job-name=vec_kmeans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=1 
#SBATCH --time=00:05:00
#SBATCH --output=test2.out
#SBATCH --constraint=amd
#SBATCH --reservation=fri
#SBATCH --propagate=STACK


FILE=k-means_sca
#FILE=example



# more about flags for amd rome 
# https://prace-ri.eu/wp-content/uploads/Best-Practice-Guide_AMD.pdf page 10 
gcc -fno-tree-vectorize -Wall ${FILE}.c  -march=znver1 -o ${FILE}.out  -lm 

# run
./${FILE}.out

rm ./${FILE}.out

# https://acl.inf.ethz.ch/teaching/fastcode/2021/slides/07-simd-avx.pdf