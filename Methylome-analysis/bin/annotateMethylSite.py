#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com
import os
import sys
import argparse
import numpy as np
import pandas as pd
from progLog import ProgLog
from myListDir import MyListDir


def mergeAnno(valuefile, annofile, outfile):
    # read contents from files
    value_base = os.path.basename(valuefile)
    anno_base = os.path.basename(annofile)
    log = ProgLog(infor='\t#### Start merging files: {} and {}'.format(value_base, anno_base))
    value = pd.read_csv(valuefile, sep="\t")
    log.progReport("read file {}".format(os.path.basename(value_base)))
    anno = pd.read_csv(annofile, sep="\t", header=None)
    anno.columns = ['Type', 'Anno', 'Chr', 'Position', 'End', 'Base', 'Base2']
    log.progReport("read file {}".format(anno_base))
    # merge two files
    out = pd.merge(anno, value, how='left', on=['Chr', 'Base', 'Position']).fillna('NA')
    out = out.drop(['End', 'Base2'], axis=1)
    col = out.columns[2:7]
    col = col.append(out.columns[:2])
    col = col.append(out.columns[7:])
    out = out.reindex(columns = col)
    log.progReport("merge files {} and {}".format(value_base, anno_base))
    # output to csv
    out.to_csv(outfile, sep="\t", index=False, header=True)
    log.endReport('to csv')


def annotateSite(infile, outfile):
    log = ProgLog(infor='\t#### Start annotating {}'.format(os.path.basename(infile)))
    annovar = '~/yangrui/softs/annovar'  # customize
    cmd = "{}/annotate_variation.pl -out {} -build hg19 {} {}/humandb/".format(annovar, outfile, infile, annovar)
    print("\t{}".format(cmd))
    os.system(cmd)
    log.progReport('annotated')


def formatConversion(infile, outfile, type):
    log = ProgLog(infor='\t#### Start analyzing {}'.format(os.path.basename(infile)))
    data = pd.read_csv(infile, sep="\t")
    log.progReport('read infile')
    if type == 'bed':
        out = convertToBed(data)
    elif type == 'avinput':
        out = convertToAvinput(data)
    log.progReport('infile to {}'.format(type))
    out.to_csv(outfile, sep="\t", index=False, header=None)
    log.endReport('to csv')


def convertToAvinput(data):
    out = pd.DataFrame(np.zeros((data.shape[0], 5)))
    out.columns = ['chr', 'start', 'end', 'allele1', 'allele2']
    out['chr'] = data['Chr']
    out['start'] = data['Position']
    out['end'] = data['Position']
    out['allele1'] = data['Base']
    out['allele2'] = data['Base']
    return(out)


def convertToBed(data):
    out = pd.DataFrame(np.zeros((data.shape[0], 6)))
    out.columns = ['chr', 'start', 'end', 'index', 'score', 'strand']
    out['chr'] = data['Chr']
    out['start'] = data['Position']-1
    out['end'] = data['Position']
    out['index'] = data.index
    out['score'] = '.'
    out['strand'] = '+'
    return(out)


def getRunList(first, step):
    inpath = sys.path[0] + '/../output'  # customize
    sample = MyListDir(path=inpath, retain='.methyl_level.txt')
    all_list = list(sample.fullfile_list)
    run_list = all_list[first*step : first*step+step]
    return(all_list, run_list)



if __name__ == '__main__':
    # input parameters
    parser = argparse.ArgumentParser("Input two parameters")
    parser.add_argument('-f', '--first', type=int, help="slice, first element, RECOMMENDED!")
    parser.add_argument('-s', '--step', type=int, help="slice, step size, RECOMMENDED!")
    parser.add_argument('-c', '--convert', type=bool, help='convert file format or not?')
    parser.add_argument('-a', '--anno', type=bool, help='anno methyl site using annovar or not?')
    parser.add_argument('-m', '--merge', type=bool, help='merge methyl file and anno file or not?')
    args = parser.parse_args()
    if all(p is None for p in [args.first, args.step]):
        parser.print_help()
        exit()
    # run list
    all_list, run_list = getRunList(args.first, args.step)
    print(">>> list:\nall: {} ({})\nselect: {} ({})"
          .format(all_list, len(all_list), run_list, len(run_list)))
    # run_list = ['../output/lambda_phage.methyl_level.txt']  # test the smallest sample
    # run in circle
    for infile in run_list:
        chr = os.path.basename(infile).split('.methyl_level.txt')[0]
        convfile = os.path.dirname(infile) + '/' + chr + '.avinput'
        annofile = os.path.dirname(infile) + '/' + chr
        outfile = os.path.dirname(infile) + '/' + chr + '.csv'
        print("\n>>> for {}\ninfile = {}\nconvfile = {}\nannofile = {}.variant_function\noutfile = {}".
              format(chr, infile, convfile, annofile, outfile))
        if args.convert:
            formatConversion(infile, convfile, 'avinput')
        if args.anno:
            annotateSite(convfile, annofile)
        if args.merge:
            mergeAnno(infile, annofile + '.variant_function', outfile)

# python bin/annotateMethylSite.py -f 0 -s 4 -c 1 -a 1 -m 1


