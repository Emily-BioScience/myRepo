#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com
import time

class ProgLog():
    def __init__(self, infor="Start ..."):
        self.infor = infor
        self.start = time.perf_counter()
        self.startReport()

    def startReport(self):
        print("\n{}".format(self.infor))

    def progReport(self, log="processing ..."):
        self.middle = time.perf_counter() - self.start
        print("\t{} in {:.2f} s.".format(log, self.middle))

    def endReport(self, log='all'):
        self.finish = time.perf_counter() - self.start
        print("\t{} Done in {:.2f} s.".format(log, self.finish))


if __name__ == '__main__':
    log = ProgLog()
    print('test start')
    log.progReport("print start")
    print('test progress')
    log.progReport("print progress")
    log.endReport()
