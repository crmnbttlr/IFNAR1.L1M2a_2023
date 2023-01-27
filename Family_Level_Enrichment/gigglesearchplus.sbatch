#!/bin/bash
# General Settings
#SBATCH -p short
#SBATCH --job-name=gigglesearch
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200gb
#SBATCH --time=00:10:00
#SBATCH --output=/OutputDir/gigglesearch.out
#SBATCH --error=/ErrorDir/gigglesearch.err

## Example usage:
## inDir=/bedDir/ \
## outDir=/giggleDir/ \
## repeatIndex=/Directory/repeats/indexed
## repeatPath=/Directory/repeats/sorted/
## sbatch --array 0-1 gigglesearchplus.sbatch

# define query files
queries=($(ls ${inDir}/*.bed | xargs -n 1 basename))

# load modules
module load samtools/1.10

# run the thing
pwd; hostname; date

echo "samtools version: "$(samtools --version)
echo "Processing peak file: "${queries[$SLURM_ARRAY_TASK_ID]}

# bgzip each bed file
echo $(date +"[%b %d %H:%M:%S] Starting bgzip...")

bgzip ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}

# run giggle search
echo $(date +"[%b %d %H:%M:%S] Starting giggle search...")

/Shares/CL_Shared/bin/giggle search -q ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.bed.gz -i ${repeatIndex} -s -g 3209286105 > ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.giggleStats.tmp

# remove full index name from output
echo $(date +"[%b %d %H:%M:%S] Removing full repeat name...")

awk '(NR>1){print $0}' ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.giggleStats.tmp \
| sed 's/.bed.gz//g' \
| sed "s|$repeatPath||g" \
> ${outDir}/${queries[SLURM_ARRAY_TASK_ID]%.bed}.giggleStats

# sort output by descending giggle score
# note: only use unsorted output as input into makegigglematrix.sbatch!
echo $(date +"[%b %d %H:%M:%S] Sorting by descending combo score...")

sort -k8,8rn ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.giggleStats > ${outDir}/${queries[$SLURM_ARRAY_TASK_ID].bed}.giggleStats.sorted

# remove temporary file
echo $(date +"[%b %d %H:%M:%S] Removing intermediate file...")

rm ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.giggleStats.tmp

# decompress bgzip'd bed files
echo $(date +"[%b %d %H:%M:%S] Decompressing bgzip'd bed files...")

bgzip -d ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bed}.bed.gz

echo $(date +"[%b %d %H:%M:%S] Done")