#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com
# import gzip
# import tarfile
import os
import sys
import time
import numpy as np
import pandas as pd
from os import listdir
from functools import reduce


def getPrefix(path, suffix):
    outlist = []
    for sample in listdir(path):
        if suffix in sample:
            if 'all' in sample:
                continue
            prefix = sample.split(suffix)[0]
            outlist.append(prefix)
    return(outlist)


def readinPath(path, prefix, suffix):
    outfiles = []
    file_num = 0
    for sample in listdir(path):
        if '-M' in sample:
            file_num += 1
            file = path + '/' + sample + '/' + prefix + suffix
            print("### readinPath: {} ({})".format(file, file_num))
            outfiles.append(file)
    return(outfiles)


def readGZfile(infile, num):
    prefix = infile.split('/')[-2]
    print("### readGZfile: {} ({})".format(prefix, num))
    data = pd.read_csv(infile, compression='gzip', sep="\t", header=None)
    data.columns = ['Chr', 'Base', 'Position', 'Context', 'Dinucleotide-context', prefix + '_methyl_level', 'o', 't']
    data = data.drop(['o', 't'], axis=1)
    return(data)


def concentrateCGmap(datafiles):
    all_data = []
    file_num = 0
    for file in datafiles:
        file_num += 1
        sample_data = readGZfile(file, file_num)
        sample_data = sample_data[sample_data[sample_data.columns[-1]] > 0]
        all_data.append(sample_data)
    outdata = reduce(lambda left,right:
                     pd.merge(left, right,
                              on=['Chr', 'Base', 'Position', 'Context', 'Dinucleotide-context'], how='outer'),
                     all_data).fillna('NA')
    print("### concentrateCGmap: {}".format(outdata.shape))
    return(outdata)


def initParam(first, step):
    # script_dir = sys.path[0]  # command line mode
    script_dir = os.getcwd()  # interactive mode
    inpath = script_dir + '/../data'
    outpath = script_dir + '/../output'
    suffix = '.CGmap.gz'
    chr_list = getPrefix(inpath + '/' + 'E-01-M', suffix)  # 'E-01-M': test sample
    run_list = chr_list[first*step:first*step+step]  # run in parallel
    print(">>> initParam: \nRunning '{}'\nInput dir: {}\nOutput dir: {}\nAll list: {} ({})\nRunning list: {} ({})\n".format(sys.argv[0], inpath, outpath,
          chr_list, len(chr_list), run_list, len(run_list)))
    return(inpath, outpath, suffix, run_list)


def runConcentrate(inpath, outpath, suffix, prefix):
    start = time.perf_counter()
    datafiles = readinPath(inpath, prefix, suffix)
    reportTime("readinPath Done", start)
    outdata = concentrateCGmap(datafiles)
    reportTime("concentrateCGmap Done", start)
    outdata.to_csv(outpath + '/' + prefix + '.methyl_level.txt', sep="\t", index=False)
    reportTime("outToCSV Done", start)


def reportTime(tag, start):
    progress = time.perf_counter()
    print("### {} in {:.4f} s\n".format(tag, progress-start))



if __name__ == '__main__':
    # first, step = eval(input("Please input the first index, and the step: "))  # control parallel
    if len(sys.argv) < 3:
        print("Two numbers required: first index, and the step!")
        exit()
    else:
        first = eval(sys.argv[1])
        step = eval(sys.argv[2])
        print(">>> Input: \nfirst: {}\nstep: {}\n".format(first, step))
    # init Parameters
    inpath, outpath, suffix, run_list = initParam(first, step)
    # run in circle
    for prefix in run_list:
        print("\n>>> Start analyzing '{}' ...".format(prefix))
        runConcentrate(inpath, outpath, suffix, prefix)
