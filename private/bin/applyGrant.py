#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com


def calcAge(birth, current):
    age = current['year'] - birth['year'] - ((current['month'], current['day']) < (birth['month'], birth['day']))
    return(age)


def calcPossibility(birth, current, limit):
    applylist = []
    age = calcAge(birth, current)
    for title, cutoff in limit.items():
        if age < cutoff:
            applylist.append(title)
    print("{} in {}.{} : {}".format(age, current['year'], current['month'], ",".join(applylist)))



if __name__ == '__main__':
    birth = {'year': 1985, 'month': 11, 'day': 17}
    
    limit = {'优秀青年': 40,
             '青年长江': 38,
             '万人拔尖': 37}

    for y in range(2020, 2025):
        start = {'year': y, 'month': 1, 'day': 1}
        end = {'year': y, 'month': 12, 'day': 30}
        calcPossibility(birth, start, limit)
        calcPossibility(birth, end, limit)


