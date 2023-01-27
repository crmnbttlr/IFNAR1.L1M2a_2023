#!/bin/bash
# General settings
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
# Job name and output
#SBATCH -J bedgraphToBigwig
#SBATCH -o /OutputDir/convert.out
#SBATCH -e /ErrorDir/convert.err


## Example usage:
## inDir=/macsDir/ \
## outDir=/bwDir/ \
## chromSizesFile=/Directory/hg.chrom.sizes
## sbatch --array 0-0 bdg_to_bw.sbatch

# define query files
queries=($(ls $inDir/*treat*.bdg | xargs -n 1 basename))

# run the thing
pwd; hostname; date

echo "Processing file: "${queries[$SLURM_ARRAY_TASK_ID]}
echo $(date +"[%b %d %H:%M:%S] Sorting bedgraph...")

sort -k1,1 -k2,2n ${inDir}/${queries[$SLURM_ARRAY_TASK_ID]} > ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bdg}.sorted.bdg

echo $(date +"[%b %d %H:%M:%S] Converting sorted bedgraph to bigwig...")

bedGraphToBigWig ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bdg}.sorted.bdg $chromSizesFile ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bdg}.bw

echo $(date +"[%b %d %H:%M:%S] Removing sorted bedgraph...")

rm ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]%.bdg}.sorted.bdg