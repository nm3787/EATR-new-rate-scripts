#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20MB
#SBATCH --time=12:00:00
#SBATCH --job-name=E_imetad
#SBATCH --output=run_%a/slurm_%A_%a.out

module load gcc/10.2.0

touch contact_plumed.dat
cd run_$SLURM_ARRAY_TASK_ID
/scratch/work/hockygroup/software/lammps-nico-test/src/lammps_plumed.sh -var temp 312 -var id $SLURM_ARRAY_TASK_ID -in ../new_input_script.lammps
