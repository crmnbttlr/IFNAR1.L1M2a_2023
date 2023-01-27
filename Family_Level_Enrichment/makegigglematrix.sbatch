#!/bin/bash
## General settings
#SBATCH -p short
#SBATCH --job-name=gigglematrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200gb
#SBATCH --time=00:10:00
#SBATCH --output=/OutputDir/gigglematrix.out
#SBATCH --error=/ErrorDir/gigglematrix.err

## Example usage:
## inDir=/giggleDir/ \
## orderFile=${inDir}/orderfile.txt \
## RefFile=file.giggleStats \
## sbatch makegigglematrix.sbatch

# Move to directory
echo "Working directory: "${inDir}
echo "Input files: "$(ls ${inDir}/*.giggleStats)

cd ${inDir}

# Get scores
echo $(date +"[%b %d %H:%M:%S] Creating table of scores from each input...")

for i in ${inDir}/*.giggleStats
do
	outputFileBaseName=`basename $i`
	cat ${i} | awk '{print $8}' > ${outputFileBaseName/.giggleStats/.giggleStats.score.tmp}
done

# Save the desired order of samples
echo $(date +"[%b %d %H:%M:%S] Assigning sample order to variable...")

sampleOrder=$(cat ${orderFile} | sed 's/.giggleStats/.giggleStats.score.tmp/g' | tr "\n" " ")

#Combine all scores within each set of files into a single file
echo $(date +"[%b %d %H:%M:%S] Combining all scores within each set of files into a single file...")

paste ${sampleOrder} > ${inDir}/allGiggleScores.tmp

# Get the list of repeats from one of the output files
echo $(date +"[%b %d %H:%M:%S] Getting the names of repeats from one of the output files...")

awk '{print $1}' ${inDir}/${RefFile} > ${inDir}/repNames.txt

# Add repeat names to giggle scores
echo $(date +"[%b %d %H:%M:%S] Adding repeat names to giggle scores...")

paste ${inDir}/repNames.txt ${inDir}/allGiggleScores.tmp > ${inDir}/annotatedGiggleScores.tmp

# add header containing sample order
echo $(date +"[%b %d %H:%M:%S] Adding header containing sample order...")

echo ${sampleOrder} | sed 's/_250bp_OCR.giggleStats.score.tmp//g' | awk '{print "repeat " $0}' | tr " " "\t" > ${inDir}/header.tmp

cat ${inDir}/header.tmp ${inDir}/annotatedGiggleScores.tmp > ${inDir}/giggleMatrix.tab

# Remove intermediate files
echo $(date +"[%b %d %H:%M:%S] Removing intermediate files...")

rm ${inDir}/allGiggleScores.tmp
rm ${inDir}/repNames.txt
rm ${inDir}/annotatedGiggleScores.tmp
rm ${inDir}/header.tmp

echo $(date +"[%b %d %H:%M:%S] Done")
