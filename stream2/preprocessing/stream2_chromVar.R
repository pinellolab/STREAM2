suppressMessages(library(chromVAR))
suppressMessages(library(Matrix))
suppressMessages(library(SummarizedExperiment))
suppressMessages(library(BiocParallel))
suppressMessages(library(data.table))

set.seed(2022)

### Get command line arguments
args <- commandArgs(TRUE)
print(args[[1]])
print(args[[2]])
print(args[[3]])
print(args[[4]])
print(args[[5]])

if(length(args)<5){
    print("Not sufficient arguments supplied.")
}else{
    for(i in 1:length(args)){
      eval(parse(text=args[[i]]))
    }
}

print(paste0('input is ', input))
print(paste0('species is ', species))
print(paste0('genome is ', genome))
print(paste0('feature is ', feature))
print(paste0('n_jobs is ', n_jobs))

if(species=='HomoSapiens'){
    species = 'Homo sapiens'
}else if(species=='MusMusculus'){
    species = 'Mus musculus'
}else{
    print("Only 'HomoSapiens' and 'MusMusculus' are supported")
    break
}
    
print(paste0('Using',n_jobs,'cores...',sep = " "))
register(MulticoreParam(n_jobs))

### Read in data files
print(paste0('Read in regions: ', input, '/region_file.bed...'))
peaks <- makeGRangesFromDataFrame(data.frame(fread(paste0(input,'/region_file.bed'),col.names=c('seqnames','start','end'))))

print(paste0('Read in samples from ', input, '/sample_file.tsv'))
samples <- data.frame(fread(paste0(input,'/sample_file.tsv'),header=FALSE))

print(paste0('Read in counts from ', input, '/count_file.mtx'))
counts = readMM(paste0(input,'/count_file.mtx'))

if(dim(counts)[1] == dim(samples)[1]){
    print('Transpose counts')
    counts = t(counts)
}
colnames(counts) <- samples[,1]

### Main Calculation
# Make RangedSummarizedExperiment
SE <- SummarizedExperiment(
  rowRanges = peaks,
  colData = samples,
  assays = list(counts = counts)
)

# Add GC bias
print('Add GC bias ...')
SE <- addGCBias(SE, genome = genome)

print('Filter peaks ...')
SE <- filterPeaks(SE, non_overlapping = TRUE)

print('Get background peaks ...')
bg <- getBackgroundPeaks(SE)

if(feature == 'kmer'){
  # compute kmer deviations
  print('k-mer counting...')   
  kmer_ix <- matchKmers(k, SE, genome = genome)
  print('Computing k-mer deviations...') 
  dev <- computeDeviations(object = SE, annotations = kmer_ix,background_peaks = bg)
}
if(feature == 'motif'){
  # compute motif deviations
  suppressMessages(library('JASPAR2020')) 
  suppressMessages(library('motifmatchr'))
  print('motif matching...')
  motifs = getJasparMotifs(species = species)    
  motif_ix <- matchMotifs(motifs, SE, genome = genome)
  print('Computing motif deviations...')
  dev <- computeDeviations(object = SE, annotations = motif_ix, background_peaks = bg)
}

devTable <- assays(dev)[["deviations"]]

print('Saving zscores...')
write.table(devTable, file = gzfile(file.path(input,"zscores.tsv.gz")), sep = "\t", quote = FALSE, row.names = TRUE, col.names=NA)
