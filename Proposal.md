# Title: Modelling of Point Patterns as a Point Process Realization and Sampling of Further Realizations

<!---
Spatial Point Process Modelling of a Family of Persistence Diagrams using Markov Chain Monte Carlo
-->

A point pattern in R^2 space will be modelled as a spatial point process. Then, the model will be used to generate further realizations of the point process. 

<!---
There are two levels of parallelization. 
1. At least N iterations of burn-in period for getting sufficiently independent realizations (shared memory).
2. Generation of M instances for the process using MCMC (independent memory).
-->

This is what I envision the pseudo-algorithm to look like:

Input: point pattern as *pattern*

1. fit point process to pattern and get model/process *parameters*

for i in range(M):

  1. initialize *new_realization* = pattern

  2. for j in range(N):

       for k in range(len(pattern)):

         A. pattern = new_realization
     
         B. propose *new_point* in lieu of pattern(k)
    
         C. compute *acceptance probability, ap*, using parameters
    
         D. if ap > rand, set pattern(k) = new_point, else pattern(k) = pattern(k)

     new_realization = pattern

Output: M realizations for the point process corresponding to the input point pattern

Loop k should be OpenMP parallelized because of requiring shared memory. I'll parallelize either loop i or j using MPI. If j, I'll fix M to 1 (get rid of it basically). If I parallize i, I'll fix N to a small number. 
