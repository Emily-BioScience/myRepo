#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com
# import gzip
# import tarfile
import os
import numpy as np
import pandas as pd
from os import listdir
from functools import reduce


def getPrefix(path, suffix):
    outlist = []
    for sample in listdir(path):
        if suffix in sample:
            prefix = sample.split(suffix)[0]
            outlist.append(prefix)
    return(outlist)


def readPath(path, prefix, suffix):
    outfiles = []
    for sample in listdir(path):
        file = path + '/' + sample + '/' + prefix + suffix
        print("### readPath: {}".format(file))
        outfiles.append(file)
    return(outfiles)


def readGZfile(infile):
    prefix = infile.split('/')[-2]
    print("### readGZfile: {}".format(prefix))
    data = pd.read_csv(infile, compression='gzip', sep="\t", header=None)
    data.columns = ['Chr', 'Base', 'Position', 'Context', 'Dinucleotide-context', prefix + '_methyl_level', 'o', 't']
    data = data.drop(['o', 't'], axis=1)
    return(data)


def concentrateCGmap(datafiles):
    all_data = []
    for file in datafiles:
        sample_data = readGZfile(file)
        all_data.append(sample_data)
    outdata = reduce(lambda left,right:
                     pd.merge(left, right,
                              on=['Chr', 'Base', 'Position', 'Context', 'Dinucleotide-context'], how='outer'),
                     all_data).fillna('NA')
    print("### concentrateCGmap: {}".format(outdata.shape))
    return(outdata)


if __name__ == '__main__':
    os.chdir('/public/noncode/users/wuyang/myRepo/Methylome-analysis/bin')
    inpath = '../data'
    outpath = '../output'
    testSample = 'E-01-M'
    suffix = '.CGmap.gz'
    chr_list = getPrefix(inpath + '/' + testSample, suffix)
    chr_list = chr_list[0:1]  # 控制并行
    for prefix in chr_list:
        datafiles = readPath(inpath, prefix, suffix)
        outdata = concentrateCGmap(datafiles)
        outdata.to_csv(outpath + '/' + prefix + '.methyl_level.txt', sep="\t", index=False)

