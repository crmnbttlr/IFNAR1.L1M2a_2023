#!/bin/bash
## General settings
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
# Job name and output
#SBATCH -J macs2_call_peaks
#SBATCH -o /OutputDir/macs.out
#SBATCH -e /ErrorDir/macs.err

## Example usage:
## inDir=/bamDir/ \
## outDir=/macsDir/ \
## controlFile=/bamDir/Input.bam \
## genome=hs \
## sbatch --array=0-0 macs2_calpeaks_PE.sbatch

# Load modules
module load python/2.7.14/MACS/2.1.1

# Define query files
queries=($(ls ${inDir}/*.bam | xargs -n 1 basename))

# Run the thing
pwd; hostname; date

echo "macs2 version: "$(macs2 --version)
echo "Target file: "${queries[$SLURM_ARRAY_TASK_ID]}
echo "Control file: "${controlFile}
echo $(date +"[%b %d %H:%M:%S] Starting macs2...")

# With Control
macs2 callpeak \
--format BAMPE \
--treatment ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]} \
--control ${controlFile} \
--pvalue 0.01 \
--name ${queries[$SLURM_ARRAY_TASK_ID]%.sorted.bam} \
--outdir ${outDir} \
-g ${genome} \
--SPMR -B \
--call-summits

# Without Control
#macs2 callpeak \
#--format BAMPE \
#--treatment ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]} \
#--pvalue 0.01 \
#--name ${queries[$SLURM_ARRAY_TASK_ID]%.sorted.bam} \
#--outdir ${outDir} \
#-g ${genome} \
#--SPMR -B \
#--call-summits

echo $(date +"[%b %d %H:%M:%S] Done!")