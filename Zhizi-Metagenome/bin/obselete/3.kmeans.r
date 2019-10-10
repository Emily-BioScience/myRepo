# infile = '../output/2019.09.selectSamples.txt'
infile = '../output/2019.09.selectSamples-withStool.txt'
# outfile = '../output/2019.09.selectSamples-cluster.txt'
outfile = '../output/2019.09.selectSamples-withStool-cluster.txt'

data <- read.table(infile, header=T, sep="\t")
cluster <- kmeans(data[, c(3, 4, 5, 6, 7, 8, 9)], 2)
data$Cluster <- cluster$cluster
write.table(data, file=outfile, sep="\t", quote=F, row.names=F, col.names=T)
sta = table(data[, c(ncol(data)-1,ncol(data))])
print(sta)
