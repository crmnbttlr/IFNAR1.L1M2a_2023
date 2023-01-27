#!/bin/bash
## General settings
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
## Job name and output
#SBATCH -J bwa
#SBATCH -o /OutputDir/bwa.out
#SBATCH -e /ErrorDir/bwa.err

## Note that this script assumes the mitochondrial genome is represented in the reference as 'chrM'

## Example usage:
## inDir=/trimmedDir/ \
## outDir=/bamDir/ \
## bwaIndex=/Directory/hg38.main.fa \
## genomeChrFile=/Directory/hg38.main.chrom.sizes \
## reportsDir=/reportsDir/ \
## sbatch --array 0-0 bwa_align_PE.sbatch

# Set constant variables
numThreads=4
nonChrM=$(cat ${genomeChrFile} | awk '{print $1}' | grep -v chrM | tr '\n' ' ')

# Load modules
module load bwa samtools

# Define query files
queries=($(ls ${inDir}/*.fastq.gz | xargs -n 1 basename | sed 's/_1.fastq.gz//g' | sed 's/_2.fastq.gz//g' | uniq))

# Define temporary directory
tmpDir=${outDir}/samtools_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Make temporary directory
pwd; hostname; date

echo "bwa version: "$(bwa)
echo "Samtools version: "$(samtools --version)

echo "Making temporary directories..."

mkdir -p ${tmpDir}

# Align reads to the reference genome
# Note that this will filter out low quality reads (-q 10), unmapped reads (-F 4)
pwd; hostname; date

echo "Aligning reads to the genome..."
echo "bwa version: "$(bwa)
echo "Samtools version: "$(samtools --version)

echo "Processing file: "${queries[$SLURM_ARRAY_TASK_ID]}
echo "Aligning to assembly: "${bwaIndex}

bwa mem \
-t ${numThreads} \
${bwaIndex} \
${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R1_trimmed.fastq.gz \
${inDir}/${queries[$SLURM_ARRAY_TASK_ID]}_R2_trimmed.fastq.gz \
> ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sam

# Generate flagstat files from unfiltered sam files
echo $(date +"[%b %d %H:%M:%S] Making flagstat file from unfiltered sam...")

samtools flagstat -@ ${numThreads} ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sam \
> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_flagstat.txt

# Convert sam files to bam files, filtering out low quality (<10) and unmapped (through '-F 4' option)
echo $(date +"[%b %d %H:%M:%S] Converting sam to unfiltered bam & sorting")

samtools view -@ ${numThreads} -Sb -q 10 -F 4 ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sam \
| samtools sort -@ ${numThreads} -T ${tmpDir} - \
> ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam

# Index unfiltered sorted bam files
echo $(date +"[%b %d %H:%M:%S] Indexing unfiltered, sorted bams...")

samtools index \
${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam \
> ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam.bai

# Filter out chrM reads & sort
echo $(date +"[%b %d %H:%M:%S] Removing chrM reads and sorting...")

samtools view -b ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam ${nonChrM} \
| samtools sort -@ ${numThreads} -T ${tmpDir} - \
> ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_filtered.sorted.bam

# Index filtered sorted bam files
echo $(date +"[%b %d %H:%M:%S] Indexing filtered, sorted bams...")

samtools index ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_filtered.sorted.bam \
> ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_filtered.sorted.bam.bai

# Generate flagstat files from filtered sorted bam files
echo $(date +"[%b %d %H:%M:%S] Making flagstat file from filtered bam...")

samtools flagstat -@ ${numThreads} ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}_filtered.sorted.bam \
> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_filtered_flagstat.txt

# Determine percentage of total reads that mapped to mitochondrial genes
echo $(date +"[%b %d %H:%M:%S] Calculating percentage of total reads mapping to mitochondrial genome...")

chrMreads=`samtools view -c ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam chrM`
totalReads=`samtools view -c ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam`
fractionMreads=`echo "100 * ${chrMreads} / ${totalReads}" | bc -l`
touch ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_chrMreadsFraction.txt
echo ${queries[$SLURM_ARRAY_TASK_ID]} >> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_chrMreadsFraction.txt
echo ${totalReads} 'total mapped reads' >> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_chrMreadsFraction.txt
echo ${chrMreads} 'mitochondrial reads' >> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_chrMreadsFraction.txt
echo ${fractionMreads} 'percentage of mitochondrial reads from total mapped reads' >> ${reportsDir}/${queries[$SLURM_ARRAY_TASK_ID]}_chrMreadsFraction.txt

# Remove intermediate files
echo $(date +"[%b %d %H:%M:%S] Removing intermediate files...")

rm ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sam
rm ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam
rm ${outDir}/${queries[$SLURM_ARRAY_TASK_ID]}.sorted.tmp.bam.bai
rm -r ${tmpDir}

echo $(date +"[%b %d %H:%M:%S] Done")

### Explanation of bwa options
### '-t <int' - specifies number of threads to use

### Explanation of samtools options
### '-@ <int>' - number of threads
### '-Sb' - autodetect input & output bam
### '-q <int>' - quality threshold (filter out reads with quality below int)
### '-F 4' - exclude unmapped reads
### '-b' - output bam
### '-c' - print only the count of matching records