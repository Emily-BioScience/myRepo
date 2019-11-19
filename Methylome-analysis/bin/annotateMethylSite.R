library(GenomicFeatures)

hg19.refseq.db <- makeTxDbFromUCSC(genome='hg19', table='refGene')

### bed => vcf
library("bedr")
data <- read.table('chrM.sorted.bed')
colnames(data) <- c('chr', 'start', 'end', 'name', 'score', 'strand')
data$chr <- as.character(data$chr)
data$start <- as.numeric(data$start)
data$end <- as.numeric(data$end)
bed2vcf(data, filename='test.vcf',  zero.based = TRUE, header = NULL, fasta = NULL)


library('VariantAnnotation')
vcf <- readVcf('test.vcf', "hg19")



  Input seems to be in bed format but chr/start/end column names are missing

Make sure the the chr column is character and the start and end positions are numbers
   your chr column is a factor!



# python => annovar format
annotate_variation.pl -out chrY.anno -build hg19 chrY.avinput  ~/yangrui/softs/annovar/humandb/



install.packages("bedr")
library(bedr)


#安装ChIPseeker包
source ("https://bioconductor.org/biocLite.R")
biocLite("ChIPseeker")
# 下载人的基因和lincRNA的TxDb对象
biocLite("org.Hs.eg.db")
biocLite("TxDb.Hsapiens.UCSC.hg19.knownGene")
biocLite("TxDb.Hsapiens.UCSC.hg19.lincRNAsTranscripts")
biocLite("clusterProfiler")
#载入各种包
library("ChIPseeker")
library(clusterProfiler)
library("org.Hs.eg.db")
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
library("TxDb.Hsapiens.UCSC.hg19.lincRNAsTranscripts")
lincRNA_txdb=TxDb.Hsapiens.UCSC.hg19.lincRNAsTranscripts
