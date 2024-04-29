#!/bin/bash
pattern_sizes=(10 100 500 1000 2000)
samples=(10 50 100 500 1000)

# Loop over point cloud sizes
for N in "${pattern_sizes[@]}"; do
    for M in "${samples[@]}"; do
        for mpi_ranks in 1 2 4 8 16 32; do

            for omp_threads in 1 2 4 8 16 32; do

                echo "${mpi_ranks} MPI ranks, ${omp_threads} threads, ${N} points, ${M} samples"
                mpirun -np $mpi_ranks ./parallel $N $M $omp_threads >> results.txt

            done
        done
    done
done
