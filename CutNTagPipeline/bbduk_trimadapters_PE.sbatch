#!/bin/bash
## General settings
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=5:00:00
#SBATCH --mem=64G
# Job name and output
#SBATCH -J bbduk_PE
#SBATCH -o /OutputDirectory/bbduk.out
#SBATCH -e /ErrorDirectory/bbduk.err


## Example usage:
## inDir=/fastqDir/ \
## outDir=/timmedDir/ \
## sbatch --array 0-1 bbduk_trimadapters_PE.sbatch

# Set constant variables
numThreads=1

# Load module
module load bbmap

# Define query files
# Removes R1/R2 to generate a unique identifier for each pair of files
queries=($(ls ${inDir}/*fastq.gz | xargs -n 1 basename | sed 's/_R1_001.fastq.gz//g' | sed 's/_R2_001.fastq.gz//g' | uniq))

# define key variables
adapterFile=/opt/bbmap/38.05/resources/adapters.fa

# Run bbduk
pwd; hostname; date

echo "bbduk version: "$(bbduk.sh --version)
echo "Processing file: "${queries[$SLURM_ARRAY_TASK_ID]}
echo $(date +"[%b %d %H:%M:%S] Running bbduk...")

bbduk.sh -Xmx4g in1=${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R1_001.fastq.gz \
in2=${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R2_001.fastq.gz \
out1=${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R1_trimmed.fastq.gz \
out2=${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R2_trimmed.fastq.gz \
ref=${adapterFile} \
ktrim=r k=31 mink=11 hdist=1 tpe tbo \
qtrim=r trimq=10 \
t=${numThreads}

echo $(date +"[%b %d %H:%M:%S] Done!")