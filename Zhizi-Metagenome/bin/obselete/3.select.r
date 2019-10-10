# Title     : TODO
# Objective : TODO
# Created by: WUY
# Created on: 2019/8/26

library(corrplot)

# cutoffs
num_cut <- 1
low_cut <- 5
high_cut <- 7

infile <- '../output/3rd-meeting.summaryTable.txt'
outfile <- '../output/3rd-meeting.allAnimals.txt'
selectfile <- paste("../output/3rd-meeting.num_cut", num_cut, ".low_cut", low_cut, '.high_cut', high_cut, ".kmeans.txt", sep="")
plotfile <- '../output/3rd-meeting.corrplot.pdf'

# infile <- '../output/2019.09.summaryTable.txt'
# outfile <- '../output/2019.09.allAnimals.txt'
# selectfile <- paste("../output/2019.09.num_cut", num_cut, ".low_cut", low_cut, '.high_cut', high_cut, ".kmeans.txt", sep="")
# plotfile <- '../output/2019.09.corrplot.pdf'


pdf(plotfile)
data <- read.table(infile, header=T)

all <- c()
select <- c()
group <- c('G',  'W',  '1',  '2',  'G1')




for (g in group) {
    group <- c()
    myset <- subset(data, data$Group == g)
    for (a in unique(myset$Animal)) {
        alt.24 <- myset[myset$Animal==a,]$ALT.24
        alt.48 <- myset[myset$Animal==a,]$ALT.48
        alt.72 <- myset[myset$Animal==a,]$ALT.72
        ast.24 <- myset[myset$Animal==a,]$AST.24
        ast.48 <- myset[myset$Animal==a,]$AST.48
        ast.72 <- myset[myset$Animal==a,]$AST.72

        if (alt.48>0 & alt.72>0) {
            rank <- apply(myset[myset$Animal==a,c(7,8,13,14)], 1, mean)
        } else if (alt.48>0) {
            rank <- apply(myset[myset$Animal==a,c(7,13)], 1, mean)
        } else if (alt.72>0) {
            rank <- apply(myset[myset$Animal==a,c(8,14)], 1, mean)
        }
        report <- data.frame(Group=g, Animal=a, alt.24=alt.24, alt.48=alt.48, alt.72=alt.72, ast.24=ast.24, ast.48=ast.48, ast.72=ast.72, rank=rank)
        group <- rbind(group, report)
    }
    group <- group[order(group$rank),]

    # 看参数相关性
    data_cor <- cor(group[,c(3,4,5,6,7,8,9)])
    corrplot(corr <- data_cor)

    # 手动选择
    low <- apply(group[,c(4,5,7,8)], 2, quantile)[2,]
    high <- apply(group[,c(4,5,7,8)], 2, quantile)[4,]
    low_animal <- group[apply(group[,c(4,5,7,8)]<=low, 1, sum)>=num_cut & group$rank<=low_cut,]  # 四个值，至少$num_cut个<=25%百分数，平均的rank小于等于$low_cut，有点问题
    high_animal <- group[apply(group[,c(4,5,7,8)]>=high, 1, sum)>=num_cut & group$rank>=high_cut,]  # 四个值，至少$num_cut个>=75%百分数，平均的rank大于等于$high_cut，有点问题


    if (nrow(low_animal)) {
        low_animal$Type <- 'l'
    }
    if (nrow(high_animal)) {
        high_animal$Type <- 'h'
    }

    select <- rbind(select, low_animal, high_animal)
    all <- rbind(all, group)
}

# k-means
cluster <- kmeans(select[,c(3,4,5,6,7,8,9)], 2)
select$cluster <- cluster$cluster

select <- select[order(select$Type, select$cluster),]
all <- all[order(all$Group, all$rank),]
write.table(select, file=selectfile, sep="\t", quote=F, row.names=F, col.names=T)
write.table(all, file=outfile, sep="\t", quote=F, row.names=F, col.names=T)




