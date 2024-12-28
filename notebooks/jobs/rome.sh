#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=ROME
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/rome/notebooks/jobs/out.log


python rome-test.py > out/test-gpt-j.log
