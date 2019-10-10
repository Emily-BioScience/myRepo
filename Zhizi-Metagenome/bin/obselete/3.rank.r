# Title     : TODO
# Objective : TODO
# Created by: WUY
# Created on: 2019/8/26

# infile <- '../output/3rd-meeting.processedData.txt'
# outfile <- '../output/3rd-meeting.summaryTable.txt'

infile <- '../output/2019.09.processedData.txt'
outfile <- '../output/2019.09.summaryTable.txt'

rankAnimal <- function(data, group, time) {
    s <- subset(data, data$Group == group & data$Time == time)
    s$ALT.rank <- rank(s$ALT)   #  * 100 / nrow(s)
    s$AST.rank <- rank(s$AST)   #  * 100 / nrow(s)
    return(s)
}

data <- read.table(infile, header=T)
group <- unique(data$Group)
output <- c()

for (g in group) {
    time0 <- rankAnimal(data, g, '0h')
    time24 <- rankAnimal(data, g, '24h')
    time48 <- rankAnimal(data, g, '48h')
    out <- rbind(time0, time24, time48)

    for (a in unique(out$Animal)) {
        set <- subset(out, out$Animal==a)
        set_mean <- apply(set[,c(5,6,7,8)], 2, mean)
        set_sd <- apply(set[,c(5,6,7,8)], 2, sd)
        report <- data.frame(Group=g, Animal=a,
                ALT.24=set$ALT[1], ALT.48=set$ALT[2], ALT.72=set$ALT[3],
                ALT.rank.24=set$ALT.rank[1], ALT.rank.48=set$ALT.rank[2], ALT.rank.72=set$ALT.rank[3],
                AST.24=set$AST[1], AST.48=set$AST[2], AST.72=set$AST[3],
                AST.rank.24=set$AST.rank[1], AST.rank.48=set$AST.rank[2], AST.rank.72=set$AST.rank[3],
                ALT.mean=set_mean[1], ALT.sd=set_sd[1],
                ALT.rank.mean=set_mean[3], ALT.rank.sd=set_sd[3],
                AST.mean=set_mean[2], AST.sd=set_sd[2],
                AST.rank.mean=set_mean[4], AST.rank.sd=set_sd[4])
        output <- rbind(output, report)
    }
}

write.table(output, file=outfile, row.names=F, col.names=T, quote=F, sep="\t")