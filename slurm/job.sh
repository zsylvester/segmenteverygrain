#!/bin/bash
#SBATCH --job-name=seg_1
#SBATCH --output=/dss/tbyscratch/0B/di54doz/seg/logs/seg_%j.out
#SBATCH --error=/dss/tbyscratch/0B/di54doz/seg/logs/seg_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=leonie.sonntag@stud-mail.uni-wuerzburg.de
#SBATCH --account=pr94no-c
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00

module load micromamba
micromamba run -n segmenteverygrain python seg_1.py
