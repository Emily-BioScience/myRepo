#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com

# import os
# import sys
# import time
# import gzip
# import tarfile
# import numpy as np
# import pandas as pd
# from functools import reduce
from os import listdir


class MyListDir():
    def __init__(self, path, retain='', remove='', search=-1):
        self.path = path
        self.retain = retain
        self.remove = remove
        self.search = search
        self.file_list = listdir(self.path)
        self.fullfile_list = list(map(lambda x : "/".join([self.path, x]), self.file_list))
        self.search_list = list(map(lambda x : x.split('/')[self.search], self.fullfile_list))
        if self.retain:
            self.retain_file_list()
        if self.remove:
            self.remove_file_list()
    def retain_file_list(self):
        if self.retain:
            remove_id = []
            for i in range(len(self.search_list)):
                if self.retain not in self.search_list[i]:
                    remove_id.append(i)
            self.remove_horse(remove_id)
    def remove_file_list(self):
        if self.remove:
            remove_id = []
            for i in range(len(self.search_list)):
                if self.remove in self.search_list[i]:
                    remove_id.append(i)
            self.remove_horse(remove_id)
    def remove_horse(self, remove_id):
        if remove_id:
            for i in sorted(remove_id, reverse=True):
                del self.file_list[i]
                del self.fullfile_list[i]
                del self.search_list[i]


if __name__ == '__main__':
    # 获取所有染色体，并通过first, step切分任务
    first, step = 0, 2
    suffix = '.CGmap.gz'
    # segment = 10000000
    global segment
    segment = 50000
    sample = MyListDir(path='../data', retain='-M', search=-1)  # 样本文件来源，自定义
    chrfiles = MyListDir(path='../data/E-01-M', retain=suffix, remove='all', search=-1)  # 染色体文件来源，自定义
    all_list = list(map(lambda x : x.split(suffix)[0], chrfiles.search_list))  # 染色体前缀列表
    run_list = all_list[first*step : first*step+step]  # 选取部分跑
    print(">>> Select chrs: \nall_list: {} ({})\nrun_list:{} ({})\n".
          format(all_list, len(all_list), run_list, len(run_list)))  # log information
