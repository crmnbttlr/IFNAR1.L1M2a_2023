#!/bin/bash
#SBATCH -p short
#SBATCH --job-name=heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=10:30:00
#SBATCH --output=/OutputDir/heatmap.out
#SBATCH --error=/ErrorDir/heatmap.err

#for blueprint
wget -i blueprinturls.txt

#for GEO
file="SRR########"
module load sra/3.0.0 samtools
fastq-dump ${file} --split-files
gzip ${file}.fastq