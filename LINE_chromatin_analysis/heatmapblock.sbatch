#!/bin/bash
#SBATCH -p short
#SBATCH --job-name=heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=23:30:00
#SBATCH --output=/OutputDir/heatmap.out
#SBATCH --error=/ErrorDir/heatmap.err

# inDir=/bwsDir/ \
# bedDir=/bedDir/ \
# outputfolder=/outDir/ \
# matrixoutput=OutFileName
# sbatch heatmapblock.sbatch

module load singularity

#compute matrix
windowLeft=10000
windowRight=10000
binSize=10
numCPU=64

singularity exec --bind /scratch /scratch/directory/singularity/deeptools-3.0.1-py35_1.img computeMatrix reference-point --referencePoint TSS -S ${inDir}/H3K9me3.bw ${inDir}/ATAC.bw ${inDir}/H3K4me1.bw ${inDir}/H3K27ac.bw -R ${bedDir}/L1s_with_H3K9me3_butno_enhMarks.bed ${bedDir}/L1s_with_H3K9me3_and_enhMarks.bed ${bedDir}/L1s_with_enhMarks_butno_H3K9me3.bed ${bedDir}/L1s_with_noMarks.bed --beforeRegionStartLength $windowLeft --afterRegionStartLength $windowRight --binSize $binSize --missingDataAsZero -o ${outputfolder}/${matrixoutput}.mat.gz -p $numCPU


#plot heat map
zMin=0
zMax=1.5
yMin=0
singularity exec --bind /scratch /scratch/directory/singularity/deeptools-3.0.1-py35_1.img plotHeatmap -m ${outputfolder}/${matrixoutput}.mat.gz --outFileName ${outputfolder}/${matrixoutput}.pdf --colorMap Blues --zMin $zMin --zMax $zMax --yMin $yMin --outFileSortedRegions ${outputfolder}/${matrixoutput}.dt.bed

