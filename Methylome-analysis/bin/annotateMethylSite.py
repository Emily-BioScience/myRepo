#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com
import os
import argparse
import numpy as np
import pandas as pd
from progLog import ProgLog
from myListDir import MyListDir


def annotateSite(infile, outfile):
    log = ProgLog(infor='>>> Start annotating {}'.format(infile))
    annovar = '~/yangrui/softs/annovar'
    cmd = "{}/annotate_variation.pl -out {} -build hg19 {} {}/humandb/".format(annovar, outfile, infile, annovar)
    print("\t{}".format(cmd))
    os.system(cmd)
    log.progReport('annotated')


def formatConversion(infile, outfile, type):
    log = ProgLog(infor='>>> Start analyzing {}'.format(infile))
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
    sample = MyListDir(path='../output', retain='.methyl_level.txt')
    all_list = list(sample.fullfile_list)
    run_list = all_list[first*step : first*step+step]
    return(all_list, run_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Input two parameters")
    parser.add_argument('-f', '--first', type=int, help="slice, first element")
    parser.add_argument('-s', '--step', type=int, help="slice, step size")
    parser.add_argument('-t', '--type', type=str, help='format, bed or avinput')
    parser.add_argument('-r', '--run', type=int, help='run format convertion or not?')
    args = parser.parse_args()
    if args.first is None or args.step is None or args.type is None or args.run is None:
        parser.print_help()
        exit()
    all_list, run_list = getRunList(args.first, args.step)
    print(">>> Input files:\nall: {} ({})\nselect: {} ({})"
          .format(all_list, len(all_list), run_list, len(run_list)))

    # run_list = ['../output/lambda_phage.methyl_level.txt']  # test the smallest sample
    for infile in run_list:
        chr = os.path.basename(infile).split('.methyl_level.txt')[0]
        outfile = os.path.dirname(infile) + '/' + chr + '.' + args.type
        if args.run:
            formatConversion(infile, outfile, args.type)
        annofile = os.path.dirname(infile) + '/' + chr + '.anno'
        annotateSite(outfile, annofile)


# python annotateMethylSite.py -f 0 -s 4 -t avinput -r 0
