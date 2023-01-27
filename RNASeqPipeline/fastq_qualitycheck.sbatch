#!/bin/bash
# General settings
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
# Job name and output
#SBATCH -J fastqc
#SBATCH -o /OutputDir/fastqc.out
#SBATCH -e /ErrorDir/fastqc.err


## Example usage:
## inDir=/fastqDir/ \
## outDir=/aqDir/ \
## sbatch --array 0-1 /fastq_qualitycheck.sbatch



# load modules
module load fastqc/0.11.5

# define query files
queries=($(ls $inDir/*.fastq.gz | xargs -n 1 basename))

# run the thing
pwd; hostname; date

echo "Processing file: "${queries[$SLURM_ARRAY_TASK_ID]}
echo $(date +"[%b %d %H:%M:%S] Starting fastqc...")

fastqc -o ${outDir} -f fastq -t 8 ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}

echo $(date +"[%b %d %H:%M:%S] Done!")