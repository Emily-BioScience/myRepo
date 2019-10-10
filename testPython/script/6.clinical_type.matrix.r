file = '../output/clinical_type.matrix.txt'
head <- read.table(file, nrow=1, sep="\t")
data <- read.table(file, skip=1, sep="\t")

