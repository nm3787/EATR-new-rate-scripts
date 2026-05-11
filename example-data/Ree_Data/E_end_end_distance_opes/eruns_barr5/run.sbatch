#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=2GB
#SBATCH --time=3:00:00
#SBATCH --job-name=Ebarr5

module purge
module load gcc/10.2.0
module load parallel/20201022

touch contact_plumed.dat
first=$(( 10*($SLURM_ARRAY_TASK_ID-1) + 1 ))
last=$(( 10*$SLURM_ARRAY_TASK_ID ))
seq $first $last | parallel cd run_{}';' /scratch/work/hockygroup/software/lammps-nico-new/src/lammps_plumed_mpi.sh -var temp 312 -var id {} -in ../new_input_script.lammps
