#!/bin/bash -l

### COMP328 (lab03): example batch script
###           usage: "sbatch run.sh"
###         purpose: to run serial and then parallel OpenMP & MPI on requested number of cores
###    restrictions: this script can only handle single nodes
### (c) mkbane, University of Liverpool (2020)

### Slight modifications for COMP328 programming assignment by Robert Szafarczyk, April 2021


# Specific course queue and max wallclock time
#SBATCH -p course -n 1 --exclusive -t 5

# Defaults on Barkla (but set to be safe)
## Specify the current working directory as the location for executables/files
#SBATCH -D ./
## Export the current environment to the compute node
#SBATCH --export=ALL

# load modules
## intel compiler
module load compilers/intel/2019u5
## intel mpi wrapper and run time
module load mpi/intel-mpi/2019u5/bin

echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"


SRC=$1
# Default, or input if supplied.
NUM_PARTICLES=20000 && (( $# > 1 )) && NUM_PARTICLES=$2
NUM_TIME_STEPS=50 && (( $# > 2 )) && NUM_TIME_STEPS=$3

EXE=${SRC%%.c}.exe
rm -f ${EXE}
echo compiling $SRC to $EXE
export numMPI=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
export maxNumMPI=40

echo compiling for $numMPI MPI processes

# mpiicc -O0 func2.c  mpi_quad.c -o $EXE && \
# mpiicc -O0 func2.c  mpi_quad-GATHER.c -o $EXE && \
icc  -O0
mpiicc -qopenmp -O0 $SRC -o $EXE && \

if test -x $EXE; then
      # set number of threads
      export OMP_NUM_THREADS=1

      mpirun -np ${numMPI} ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
    #   mpirun -np ${numMPI} ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
    #   mpirun -np ${numMPI} ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
else
     echo $SRC did not built to $EXE
fi
