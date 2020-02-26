#!/bin/bash
#SBATCH -J scalingtest
#SBATCH -o scalingtest.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -N 128          # Total number of nodes
#SBATCH -n 1024            # Total number of mpi tasks requested
#SBATCH -p normal  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 04:00:00     # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A A-ccsc
#SBATCH --mail-type=end
#SBATCH --mail-user=eijkhout@tacc.utexas.edu

for node in 1 2 4 8 16 32 64 128 ; do
  nlocal=100000
  export ranks=$((node*8))
  echo "Running on ${node} nodes, ${ranks} ranks"
  ibrun -np ${ranks} ./petsccg -nlocal ${nlocal} -pc_type none -ksp_max_it 10 -log_summary
done
