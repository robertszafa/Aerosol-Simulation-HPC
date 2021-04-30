#!/bin/bash -l

###         COMP328: example batch script
###           usage: "sbatch run.sh -c <num_cores_required> <sourcefile.c>"
###         purpose: to OpenMP on requested number of cores
###    restrictions: this script can only handle single nodes
### (c) mkbane, University of Liverpool (2020-2021)

### Slight modifications for COMP328 programming assignment by Robert Szafarczyk, April 2021


# Specific course queue
#SBATCH -p course -n 1 -t 5
##--exclusive

# Defaults on Barkla (but set to be safe)
## Specify the current working directory as the location for executables/files
#SBATCH -D ./
## Export the current environment to the compute node
#SBATCH --export=ALL

# load intel compiler
module load compilers/intel

if [[ $# -lt 1 ]]; then
    echo
    echo ERROR
    echo $0 -c NUM_THREADS_TO_USE NAME_OF_INPUT.c [numAerosolParticles, numTimeSteps]
    exit -1
fi

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

icc -qopenmp -O0 $SRC -o $EXE
# icc -qopenmp-stubs -O0 $SRC  -o $EXE

export OMP_PROC_BIND=TRUE
export OMP_DYNAMIC=FALSE

if test -x $EXE; then
      # set number of threads
      export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
      echo using ${OMP_NUM_THREADS} OpenMP threads
      # run multiple times
      ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
      ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
      ./${EXE} $NUM_PARTICLES $NUM_TIME_STEPS;echo
else
     echo $SRC did not built to $EXE
fi
