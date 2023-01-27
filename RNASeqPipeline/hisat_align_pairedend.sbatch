#!/bin/bash
# General Settings
#SBATCH -p short
#SBATCH --job-name=align_hisat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60gb
#SBATCH --time=24:00:00
#SBATCH --output=/OutputDir/align.out
#SBATCH --error=/Error/align.err

#example usage
#inDir=/fastqDir/ \
#outDir=/hisatDir/ \
#hisatIndexDir=/Directory/hg38/index/hisat2
#hisatIndex=hg38.main
#sbatch --array 0-1 hisat_align_pairedend.sbatch

numThreads=20

queries=($(ls ${inDir}/*.fastq.gz | xargs -n 1 basename | sed 's/_R1_001.fastq.gz//g' | sed 's/_R2_001.fastq.gz//g' | sort | uniq))

module load hisat2 samtools

hisat2 -p ${numThreads} --no-softclip -x ${hisatIndexDir}/${hisatIndex} \
-1 ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R1_001.fastq.gz \
-2 ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R2_001.fastq.gz \
| samtools view -q 10 -Sb - \
| samtools sort -@ ${numThreads} - -o ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.uniq.bam

samtools index ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.uniq.bam
