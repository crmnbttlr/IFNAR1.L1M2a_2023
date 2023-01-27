#!/bin/bash
# General Settings
#SBATCH -p short
#SBATCH --job-name=bamcoverage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --output=/OutputDir/bamcoverage.out
#SBATCH --error=/ErrorDir/bamcoverage.err

#example usage
#inDir=/hisatDir/ \
#outDir=/bigwigDir/ \
#sbatch --array 0-1 deeptools_bamcoverage_stranded.sbatch

module load singularity

deeptools=/scratch/Shares/public/singularity/deeptools-3.0.1-py35_1.img

queries=($(ls $inDir/*.sorted.uniq.bam | xargs -n 1 basename))

binSize=1
numCPU=16

singularity exec --bind /scratch ${deeptools} \
bamCoverage \
-b ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]} \
-o ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]/.sorted.uniq.bam/.fwd.bw} \
--binSize=${binSize} \
-p ${numCPU} \
--normalizeUsing CPM \
--filterRNAstrand forward \
--ignoreForNormalization chrX chrM

singularity exec --bind /scratch ${deeptools} \
bamCoverage \
-b ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]} \
-o ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]/.sorted.uniq.bam/.rev.bw} \
--binSize=${binSize} \
-p ${numCPU} \
--normalizeUsing CPM \
--filterRNAstrand reverse \
--ignoreForNormalization chrX chrM