# -*- coding UTF-8 -*-
from bs4 import BeautifulSoup

def decodeFile(infile, outfile):
    # soup = BeautifulSoup(open(infile), "html.parser")
    soup = BeautifulSoup(open(infile), "html5lib")
    # print(soup.td)  # 查询所有内容中第一个符合要求的标签

    data_list = []
    for i, tr in enumerate(soup.find_all('tr')):
        if i != 0:
            tds = tr.find_all('td')
            data_list.append({
                '序号': tds[0].contents[0],
                '部门': tds[1].contents[0],
                '姓名': tds[2].contents[0],
                '现任属性': tds[3].contents[0],
                '现任岗位': tds[4].contents[0],
                '任职时间': tds[5].contents[0],
                '申报岗位': tds[6].contents[0]
            })
    print(data_list)

    fo = open(outfile, 'w')
    for d in data_list:
        name = d['姓名']
        result = 'y' if name in accept else 'n'
        aim = d['申报岗位']

        if aim == '副研究员三级' or aim == '高级工程师三级':
            out = "\t".join([d['序号'], d['部门'], d['姓名'], d['现任属性'], d['现任岗位'], d['任职时间'], d['申报岗位'], result])
            fo.writelines([out, "\n"])




if __name__ == '__main__':
    accept = ['叶靖', '杜子东', '李文明', '肖俊敏', '闵巍庆', '敖翔', '彭晓晖', '焦臻桢', '于雷', '孙刚', '李超', '张文力', '罗韬', '赵莹', '薛源海']
    decodeFile('../data/pingpin2018.txt', '../output/pingpin2018.table.txt')