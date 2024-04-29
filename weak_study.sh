#!/bin/bash
pattern_sizes=(30 60 120 240 480 960)
samples=(30 60 120 240 480 960)

# Loop over point cloud sizes
for N in "${pattern_sizes[@]}"; do
    for M in "${samples[@]}"; do
        for mpi_ranks in 1 2 4 8 16 32; do

            for omp_threads in 1 2 4 8 16 32; do

                echo "${mpi_ranks} MPI ranks, ${omp_threads} threads, ${N} points, ${M} samples"
                mpirun -np $mpi_ranks ./parallel $N $M $omp_threads >> weak_study.txt

            done
        done
    done
done
