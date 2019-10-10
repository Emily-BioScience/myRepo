# -*- coding UTF-8 -*-
import pandas as pd
import os.path as op
import rank_modules as rankm


if __name__ == '__main__':
    # 2019.09.26 第二次实验
    infile = '../data/2019.09.rawData.txt'
    removefile = '../data/2019.09.removeSample.txt'   # 去除没有粪便组
    tablefile = '../output/2019.09.summaryTable-withStool.txt'
    clusterfile = '../output/2019.09.selectSamples-withStool-kmeans.txt'
    figurefile = '../output/2019.09.selectSamples-withStool.kmeans.jpg'
    data = pd.read_csv(infile, sep="\t")
    data.columns = ['Animal', 'Time', 'ALT', 'AST', 'TBil']  # 新版本，添加TBil
    ALT = rankm.extractFeature(data, 'ALT')
    AST = rankm.extractFeature(data, 'AST')
    TBil = rankm.extractFeature(data, 'TBil')  # 新版本，添加TBil
    summary = pd.merge(ALT, AST, how='left', on=['Group', 'Animal'])
    summary = pd.merge(summary, TBil, how='left', on=['Group', 'Animal'])  # 新版本，添加TBil
    if op.exists(removefile) and op.isfile(removefile):
        summary = rankm.removeSample(summary, removefile)
    summary.to_csv(tablefile, sep="\t", index=False)
    summary = rankm.calcMeanRank(summary)  # remove missing values and calculate mean rank
    summary = rankm.selectAnimals(summary)  # select by both value and rank
    summary = rankm.kmeans_clustering(summary, figurefile)  # K-means clustering, PCA and plotting
    summary.to_csv(clusterfile, sep="\t", index=False)