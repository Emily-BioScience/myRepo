# -*- coding: UTF-8 -*-
import jieba


# 集合元素类型，元素惟一，无序，不可变，大括号和逗号组成
def myset_func():
    A = {'python', 123, ('python', 123)}
    print(type(A))
    print("A: {}".format(A))

    B = set("python123")
    print("B: {}".format(B))

    C = {'python', 'python', 123, 234}
    print("C: {}".format(C))

    # 集合运算，集合运算的结果可以赋值给新的变量
    print("\nA|C: {}".format(A|C))  # 求并集，或者在B，或者在C，全部拿过来
    print("A-C: {}".format(A-C))  # 包括在B中，但不包括C
    print("A&C: {}".format(A&C))  # 求交集
    print("A^C: {}".format(A^C))  # 求补集，B和C之中，不相同的元素
    print("A>=C: {}".format(A>=C))  # 看C是否是A的子集
    print("A>C: {}".format(A>C))  # 看C是否是A的真子集，包含关系
    print("A<=C: {}".format(A<=C))  # 看A是否是C的子集
    print("A<C: {}".format(A<C))  # 看A是否是C的真子集，包含关系

    # 集合增加运算，更新左侧集合
    A = {'python', 123, ('python', 123)}
    A |= C
    print("\nA|=C: {}".format(A))  # 求并集，或者在B，或者在C，全部拿过来，更新A

    A = {'python', 123, ('python', 123)}
    A -= C
    print("A-=C: {}".format(A))  # 包括在B中，但不包括C，更新A

    A = {'python', 123, ('python', 123)}
    A &= C
    print("A&=C: {}".format(A))  # 求交集，更新A

    A = {'python', 123, ('python', 123)}
    A ^= C
    print("A^=C: {}".format(A))  # 求补集，B和C之中，不相同的元素，更新A

    # 集合处理方法
    A.add('test')  # 添加元素，如果不含的话
    print("\nadd A: {}".format(A))
    A.discard('test')  # 删除元素
    print("discard A: {}".format(A))
    A.add('test')  # 添加元素，如果不含的话
    A.remove('test')  # 删除元素，如果不含的话，会报错
    print("remove A: {}".format(A))
    A.clear()  # 清空集合
    print("clear A: {}".format(A))
    A.add('test')  # 添加元素，如果不含的话
    print("add A: {}".format(A))
    popout = A.pop()  # 随机返回一个元素，若为空集，则报错
    print("pop A: {}".format(popout))

    # 元素拷贝
    D = C.copy()
    setlen = len(D)
    print("D: {}".format(D))
    print("len: {}".format(setlen))

    # 元素判断
    if 123 in D:
        print(123, 'in D')
    if '234' not in D:
        print(234, 'not in D')

    try:
        while True:
            print(D.pop(), end=",")
    except:
        pass

    # 集合类型的典型应用场景，数据去重
    mylist = [1, 3, 3, 5, 'p', 'p', 'y']
    print("\noriginal list:", mylist)
    myset = set(mylist)
    print("nr list:", list(myset))


# 序列类型，有序，可通过下标访问，是基类，包括字符串类型，元组类型，和列表类型
def mylist_func():
    mylist = [1, 3, 3, 5, 'p', 'p', 'y']
    for i in mylist:
        print(i)

    # 判断
    if 'y' in mylist:
        print('y', 'in mylist')
    if 'x' not in mylist:
        print('x', 'not in mylist')

    # 操作符
    mylist2 = ['t', 'e', 's', 't']
    print("\nmylist + mylist2 =", mylist + mylist2)  # 连接
    print("mylist * 2 =", mylist*2)  # 复制
    print("mylist2[2]:", mylist2[2])  # 索引
    print("mylist2[1:5:2]:", mylist2[1:5:2])  # 以2为步长，切片
    print("mylist2[::-1]:", mylist2[::-1])  # 将列表元素取反

    # 通用函数
    print("\nlen(mylist2):", len(mylist2))  # 求长度
    print("max(mylist2):", max(mylist2))  # 求最大值
    print("min(mylist2):", min(mylist2))  # 求最小值
    print("mylist2.index('t'):", mylist2.index('t'))  # 返回第一次出现't'元素的下标
    print("mylist2.index('t', 1, 5):", mylist2.index('t', 1, 5))  # 返回第一次出现't'元素的下标, 下标1-5之间
    print("mylist2.count('t'):", mylist2.count('t'))  # 返回出现't'元素的次数

    # 元组类型：序列类型，创建后不可更改，小括号，或tuple()来创建
    creature = ('dog', 'cat', 'horse')
    color = ('human', creature)
    print("\n", type(creature))
    print(creature)
    print(type(color))
    print(color)
    print(creature[::-1])
    print(color[-1][2])

    # 列表类型：序列类型，创建后可以更改，中括号，或list()来创建
    ls = ['haha', 'test', 'python', 123]
    lt = ls  # 此时并没有在系统是重新生成一个列表，只不过弄了个别名，指针指向是一样的，但方括号和list()是真正的在创建列表
    print("\n", ls)
    ls[2] = 3  # 通过索引直接赋值
    print(ls)
    ls[1:5:2] = lt[0:2]   # 切片产生的子列表，直接替换
    print(ls)
    del ls[1:5:2]  #  切片产生的子列表，直接删除  !!!!!!!!!!
    print(ls)
    ls += lt  # 更新列表，直接相加
    print(ls)

    # 增删改查
    ls = ['haha', 'test', 'python', 123]
    ls.append('234')  # 增加元素
    ls += ['t', 'p']
    print("\nappend 234:", ls)
    ls.clear()  # 元素清空
    print("clear:", ls)
    ls = ['haha', 'test', 'python', 123]
    lt = ls.copy()  # 列表复制
    print("copy:", lt)
    ls.insert(3, 'wow')  # 插入新元素，在索引后面
    print("insert:", ls)
    ls.pop(3)  # 删除该索引处元素
    print("pop:", ls)
    ls.remove('test')   # 删除第一个出现的x元素
    print("remove:", ls)
    ls.reverse()  # 列表反转
    print("reverse:", ls)


# 字典类型，映射的体现，用{}或者dict()来创建，无序，集合类型也用{}，但没有键值对，因此，空集合得用set()来表示
def mydict_func():
    mydict = {'A':'123', 'B':'234'}
    print(mydict['A'])
    emptydict = {}  # 大括号生成空字典
    print(type(emptydict))

    # 删除字典中某一键值对
    print(mydict)
    del mydict['A']
    print(mydict)

    # 判断键是否属于该字典
    if 'A' in mydict:
        print(mydict['A'])
    else:
        print("no 'A'")

    # 返回字典的键、值、键值对，可以用来做遍历，但不能当作列表来使用
    mydict['C'] = 'test'
    mydict['D'] = 'haha'
    for k in mydict.keys():
        print("keys:", k)
    for v in mydict.values():
        print("values:", v)
    for k, v in mydict.items():
        print("keys:", k, ", values:",v)

    # d.get()和d.pop()，取出或删除字典中某一键值对，如果不存在，则返回自选的default值
    print(mydict)
    print('get A', mydict.get('A', 'no value'))  # 取出键A对应的值，如果没有，则返回所选的default值
    print(mydict)
    print('pop A', mydict.pop('A', 'no value'))  # 删除键A对应的值，如果没有，则返回所选的default值
    print(mydict)
    print('pop B', mydict.pop('B', 'no value'))  # 删除键A对应的值，如果没有，则返回所选的default值
    print(mydict)

    # d.popitem()，len(), d.clear()
    print('popitem random', mydict.popitem())  # 随机删某一键值对，返回键值对元组
    print(mydict)
    mydict['C'] = 'test'
    mydict['D'] = 'haha'
    print(mydict)
    print(len(mydict))
    print(mydict.clear())
    print(mydict)
    print(len(mydict))

    # test
    d = {}
    d['a'] = 1
    d['b'] = 3
    d['b'] = 2
    if 'C' in d.keys():
        print(d['C'])
    else:
        print("no key: 'C'")
    print(len(d))
    d.clear()


# 读取文件，并进行噪音处理，得到归一化文本
def getText(file):
    content = []
    fh = open(file, 'r')
    for line in fh:
        txt = line.strip()
        txt = txt.lower()
        for ch in '（），。： (),.: ':
            txt = txt.replace(ch, "")   # 去除乱七八糟的字符
        content.append(txt)
    return(content)


def use_jieba(file):
    # 准备变量，读文件
    count = 0
    key = {}
    detail = {}
    content = getText(file)
    # jieba.add_word('稽留流产')  # 给分词词典增加新词
    excludes = {'去年','中途'}  # 反复运行代码，人工去除不相关关键词，不断优化、修改

    # 按行分词，存入变量中
    for line in content:
        detail[count] = {}
        words = jieba.lcut(line)  # 精确模式
        # words = jieba.lcut(line, cut_all=True)  # 全模式
        # words = jieba.lcut_for_search(line)  # 搜索引擎模式

        for w in words:
            key[w] = key.get(w, 0) + 1
            detail[count][w] = detail[count].get(w, 0) + 1
        count += 1

    # 去除不相关关键词
    for e in excludes:
        del key[e]

    # 对关键词进行排序
    items = list(key.items())
    items.sort(key=lambda x:x[1], reverse=True)  # 按总出现频率从大往小排
    print("No. of keys:", len(items))
    print("No. of lines:", count)

    # 控制输出的行列数
    keyNum = 800   # 列数
    # count = 10  # 行数
    fo = open('output/1.6.clinical_type.matrix.txt', 'w')

    # 文件头一行，是关键词的信息
    header = "ID"
    for i in range(keyNum):
        key, all_freq = items[i]
        header = header + "\t" + key
    fo.write("{}\n".format(header))

    # 文件其余行，是每一个病人ID，在每个关键词上的对应频率
    for line_num in range(count):
        line = str(line_num)
        for i in range(keyNum):  # 取前500个关键词，编码特征向量
            key, count = items[i]
            line_freq = detail[line_num].get(key, 0)
            line = line + "\t" + str(line_freq)
        fo.write("{}\n".format(line))



if __name__ == '__main__':
    # myset_func()
    # mylist_func()
    # mydict_func()

    use_jieba('data/1.6.clinical_diagnose.subtypes.txt')
