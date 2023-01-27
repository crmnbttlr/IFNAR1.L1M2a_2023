#!/bin/bash
# General Settings
#SBATCH -p short
#SBATCH --job-name=featurecounts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --output=/OutputDir/counts.out
#SBATCH --error=/ErrorDir/counts.err

##example usage
##inDir=/hisatDir/ \
##outDir=/featureCountsDir/ \
##annotationFile=/Directory/gencode.v39.annotaion.gtf
##sbatch featureCounts.sbatch

numThreads=8

module load subread

inputBam=${inDir}/*.bam

featureCounts \
-p \
-O \
-T ${numThreads} \
-s 0 \
-t exon \
-g gene_id \
-a ${annotationFile} \
-o ${outDir}/featureCounts_s0.txt \
${inputBam}

featureCounts \
-p \
-O \
-T ${numThreads} \
-s 1 \
-t exon \
-g gene_id \
-a ${annotationFile} \
-o ${outDir}/featureCounts_s1.txt \
${inputBam}

featureCounts \
-p \
-O \
-T ${numThreads} \
-s 2 \
-t exon \
-g gene_id \
-a ${annotationFile} \
-o ${outDir}/featureCounts_s2.txt \
${inputBam}
