# -*- coding UTF-8 -*-
import numpy as np
import pandas as pd
import os.path as op
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# dataframe重塑，一个动物样本一行，每个feature按3个时间点分三列
def extractFeature(data, feature):
    # extract each feature
    sub = data.pivot(index='Animal', columns='Time', values=feature)
    sub.columns = feature + '.' + sub.columns

    # add animal name from index information
    newcol = sub.columns.insert(0, 'Animal')
    sub = sub.reindex(columns=newcol, fill_value=0)
    sub['Animal'] = sub.index

    # add group name from animal information
    newcol = sub.columns.insert(0, 'Group')
    sub = sub.reindex(columns=newcol, fill_value=0)
    sub['Group'] = np.array(pd.Series(sub.index).str.split('-', expand=True)[0])

    # remove index name
    sub.index = range(sub.index.shape[0])
    return(sub)


# 根据removefile里的list，去除72h无粪便组
def removeSample(summary, removefile):
    retain = summary
    remove = pd.read_csv(removefile)
    remove.columns = ['Animal']  # 72h，没有粪便样本

    empty = pd.DataFrame(np.zeros((remove.shape[0], retain.shape[1])))
    empty.columns = retain.columns
    empty['Animal'] = remove['Animal']
    remove = pd.merge(empty, remove)

    # remove samples
    retain = retain.append(remove)
    retain = retain.append(remove)
    retain = retain.drop_duplicates(subset=['Animal'], keep=False)
    return(retain)


# 按组求rank，并根据48h和72h的AST、ALT值，计算平均rank
def calcMeanRank(summary):
    # remove missing values
    f = summary.columns[[3, 4, 6, 7]]  # 只看ALT和AST，只看48h和72h
    subset = summary[(summary[f[0]]>0) & (summary[f[1]]>0) & (summary[f[2]]>0) & (summary[f[3]]>0)]
    print("Remove {} animals with missing values.".format(summary.shape[0] - subset.shape[0]))

    # add columns for ranking
    empty = pd.DataFrame(np.zeros((subset.shape[0], subset.shape[1]-1)))
    empty.columns = subset.columns[1:]
    empty.columns = [c.replace('.', '.rank.') for c in empty.columns]
    empty['Animal'] = subset['Animal']
    new = pd.merge(subset, empty, how='left', on=['Animal'])

    # rank for each feature
    for i in range(2, summary.columns.size):  # 0: Group: 1: Animal: 2-n: data
        colname = subset.columns[i]
        colrank = colname.replace('.', '.rank.')
        new[colrank] = new[colname].groupby(new['Group']).rank()

    # calculate mean rank
    addcol = new.columns.insert(new.columns.size, 'Mean_rank')
    new = new.reindex(columns=addcol, fill_value=0)
    r = [c.replace('.', '.rank.') for c in f]  # 只看ALT和AST，只看48h和72h
    new.loc[:, 'Mean_rank'] = new[r].mean(1)
    return(new)


# 按value和mean_rank，选择实验样本
def selectAnimals(summary):
    # 选择的cutoffs，及入组的组别
    low_cut = 0.35
    high_cut = 0.6
    num_cut = 1
    group = ['G', 'W', '1', '2', 'G1']  # 计入考量的组别，

    # add new columns
    empty = pd.DataFrame(np.zeros((summary.shape[0], 9)))
    empty.columns = ['Animal', 'alt48q', 'alt72q', 'ast48q', 'ast72q', 'ValueLow', 'ValueHigh', 'Percent', 'Type']
    empty.index = summary.index
    empty['Animal'] = summary['Animal']
    new = pd.merge(summary, empty, how='left', on=['Animal'])

    # calcuate merics
    low = new.groupby(new['Group']).quantile(low_cut)
    high = new.groupby(new['Group']).quantile(high_cut)
    count = new.groupby(new['Group']).count()

    # 为每个动物分别做选择
    for i in new.index:
        line = new.ix[i]
        g, name, alt48, alt72, ast48, ast72, rank = line[[0, 1, 3, 4, 6, 7, -9]]
        if g in group:  # 关心的组别
            alt48q =[low[new.columns[3]].ix[g], high[new.columns[3]].ix[g]]  # 分组计算25%和75%百分位数
            alt72q =[low[new.columns[4]].ix[g], high[new.columns[4]].ix[g]]  # 分组计算25%和75%百分位数
            ast48q =[low[new.columns[6]].ix[g], high[new.columns[6]].ix[g]]  # 分组计算25%和75%百分位数
            ast72q =[low[new.columns[7]].ix[g], high[new.columns[7]].ix[g]]  # 分组计算25%和75%百分位数

            data = {'feature': ['alt48', 'alt72', 'ast48', 'ast72'],
                    'value': [alt48, alt72, ast48, ast72],
                    'low': [alt48q[0], alt72q[0], ast48q[0], ast72q[0]],
                    'high': [alt48q[1], alt72q[1], ast48q[1], ast72q[1]]}
            data = pd.DataFrame(data)

            new.loc[i, 'alt48q'] = "{:.1f}-{:.1f}".format(alt48q[0], alt48q[1])
            new.loc[i, 'alt72q'] = "{:.1f}-{:.1f}".format(alt72q[0], alt72q[1])
            new.loc[i, 'ast48q'] = "{:.1f}-{:.1f}".format(ast48q[0], ast48q[1])
            new.loc[i, 'ast72q'] = "{:.1f}-{:.1f}".format(ast72q[0], ast72q[1])
            new.loc[i, 'ValueLow'] = sum(data['value']<data['low'])  # 计算4个值里，有几个小于25%分位数
            new.loc[i, 'ValueHigh'] = sum(data['value']>data['high'])  # 计算4个值里，有几个大于75%分位数
            new.loc[i, 'Percent'] = rank / count[count.columns[0]][g]  # 将rank转化到100%尺度

            # assign type
            if new.loc[i]['ValueLow'] > num_cut and new.loc[i]['ValueHigh'] == 0 and  new.loc[i]['Percent'] < low_cut: # 至少num_cut个值小于各自的25%分位数，无值大于75%分位数，且Mean_rank小于low_cut
                new.loc[i, 'Type'] = 'l'
            elif new.loc[i]['ValueHigh'] > num_cut and new.loc[i]['ValueLow'] == 0 and new.loc[i]['Percent'] > high_cut:  # 至少num_cut个值大于各自的25%分位数，无值小于25%分位数，且Mean_rank大于high_cut
                new.loc[i, 'Type'] = 'h'

        else:
            print("skip line: {} {}.".format(g, name))
            continue

    # select and sort data
    new = new[new['Type'] != 0]  # 去除skipped lines
    new = new.sort_values(by=['Type', 'Group'], ascending=False)
    return(new)


def kmeans_clustering(summary, outfig):
    # kmeans clustering
    data = summary[summary.columns[[2, 3, 4, 5, 6, 7]]]
    n_samples, n_features = data.shape
    print("n_samples: {}, \tn_features: {}\n".format(n_samples, n_features))
    estimator = KMeans(n_clusters = 2)
    y = estimator.fit_predict(data)
    center = estimator.cluster_centers_
    print("label: {}\ncenter: {}\n".format(y, center))
    # add columns
    newcol = summary.columns.insert(summary.columns.size, 'Cluster')
    summary = summary.reindex(columns = newcol, fill_value = 0)
    summary['Cluster'] = y
    # plot clustering results
    pca = PCA(n_components = 2)
    x = pd.DataFrame(pca.fit_transform(data))
    x2 = pd.DataFrame(pca.fit_transform(center))
    # x['Label'] = np.array(summary['Type'].str.cat(summary['Animal'], sep=', '))
    x['Label'] = np.array(summary['Type'])
    for i in x.index:
        if x.ix[i]['Label'] == 'l':
            x.loc[i, 'Label'] = None
    x['Cluster'] = y

    plt.figure()
    plt.scatter(x[0], x[1], c=x['Cluster'])
    for i in x.index:
        line = x.ix[i]
        plt.annotate(line['Label'], xy=(line[0], line[1]), xytext=(line[0], line[1]+20))
    plt.plot(x2[0], x2[1], 'r^')
    # plt.show()
    plt.savefig(outfig, dpi=600)  # PNG文件
    return(summary)



if __name__ == '__main__':
    # 2019.09.26 第二次实验
    infile = '../data/2019.09.rawData.txt'
    removefile = '../data/2019.09.removeSample.txt'   # 去除没有粪便组
    tablefile = '../output/2019.09.summaryTable-withStool.txt'
    clusterfile = '../output/2019.09.selectSamples-withStool-kmeans.txt'
    figurefile = '../output/2019.09.selectSamples-withStool.kmeans.jpg'
    data = pd.read_csv(infile, sep="\t")
    data.columns = ['Animal', 'Time', 'ALT', 'AST', 'TBil']  # 新版本，添加TBil
    ALT = extractFeature(data, 'ALT')
    AST = extractFeature(data, 'AST')
    TBil = extractFeature(data, 'TBil')  # 新版本，添加TBil
    summary = pd.merge(ALT, AST, how='left', on=['Group', 'Animal'])
    summary = pd.merge(summary, TBil, how='left', on=['Group', 'Animal'])  # 新版本，添加TBil
    if op.exists(removefile) and op.isfile(removefile):
        summary = removeSample(summary, removefile)
    summary.to_csv(tablefile, sep="\t", index=False)
    summary = calcMeanRank(summary)  # remove missing values and calculate mean rank
    summary = selectAnimals(summary)  # select by both value and rank
    summary = kmeans_clustering(summary, figurefile)  # K-means clustering, PCA and plotting
    summary.to_csv(clusterfile, sep="\t", index=False)
