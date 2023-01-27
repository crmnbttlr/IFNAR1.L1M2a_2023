#!/bin/bash
#SBATCH -p short
#SBATCH --job-name=heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=10:30:00
#SBATCH --output=/Output/download.out
#SBATCH --error=/Error/download.err

wget -i blueprinturls.txt