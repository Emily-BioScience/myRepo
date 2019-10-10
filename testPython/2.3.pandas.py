# -*- coding UTF-8 -*-
import numpy as np
import pandas as pd


def usePandasSeries():
    # series类型，与各种数据类型的兼容性不错，分为index和values两部分，操作类似ndarray类型，也类似python的字典类型
    s = pd.Series([9, 8, 7, 6])  # 从python列表开始创建
    print("series: ", s)
    s2 = pd.Series([9, 8, 7, 6], index=['a', 'b', 'c', 'd'])  # 自定义索引
    print("series 2: ", s2)
    s3 = pd.Series(25, index=['a', 'b', 'c'])  # index=，不能省略，值都是25，从标量开始创建，index表达series类型的尺寸
    print("series 3: ", s3)
    s4 = pd.Series({'a':9, 'b':8, 'c':7})  # 从字典开始创建， index=，可以指定顺序
    print("series 4: ", s4)
    s5 = pd.Series(np.arange(5))  # 从ndarray开始创建
    print("series 5: ", s5)

    # series类型，包含index和values两部分，保留numpy类型，又建了一个index类型，两者结合，就是pd.Series类型
    print("index: ", s5.index)  # index类型，自动索引和自定义索引并存，但不能混用
    print("values: ", s5.values)  # numpy类型，索引方法相同，numpy中运算和操作也可用于Series类型，可以通过自动索引，进行切片

    # series类型，包含索引和值，如果对series进行运算和切片，获得的还是series类型，如果只选一个值，则返回的是值，不是series类型
    print("type(s[:3]): ", type(s[:3]))
    print("type(s[3]): ", type(s[3]))

    # 使用in保留字
    print("s2['b']: ", s2['b'])
    if 'b' in s2:  # 判断的是，是否在s2的索引中，不会判断自动索引，只会判断自定义索引
        print("'b' in s2: ", True)

    # 使用get方法
    test = s2.get('b', 0)
    print("s2.get('b', 0): ", test)
    test = s2.get('e', 0)
    print("s2.get('e', 0): ", test)

    # series类型的对齐操作
    a = pd.Series([1, 2, 3], ['a', 'b', 'c'])
    b = pd.Series([4, 5, 6, 7], ['b', 'c', 'd', 'e'])
    print("a: ", a)
    print("b: ", b)
    print("a+b: ", a+b)  # 按关键词进行运算，如果有nan，则运算结果是nan

    # series对象和索引都可以有一个名字，存储在属性.name中
    a.name = 'test'
    a.index.name = 'test_id'

    d = pd.DataFrame(np.random.rand(3, 2))
    print(d.cumsum())


def usePandasDataFrame():
    d = pd.DataFrame(np.arange(10).reshape(2, 5))  # 从二维ndarray创建，索引是从0开始的整数，原始的数据，增加了横向和纵向的索引，2维结构
    print("d: ", d)
    print("d.index: ", d.index)
    print("d.columns: ", d.columns)

    dt = {"one": pd.Series([1, 2, 3], index={'a', 'b', 'c'}),
          "two": pd.Series([4, 5, 6, 7], index={'d', 'e', 'f', 'g'})}
    d2 = pd.DataFrame(dt)
    print("d2: ", d2)
    print("d2.index: ", d2.index)
    print("d2.columns: ", d2.columns)

    d3 = pd.DataFrame(dt, index=['a', 'b', 'c'], columns=['two', 'three'])  # 数据会根据行列索引自动补齐
    print("d3: ", d3)

    dl = {"one": [1, 2, 3, 4, 5], "two": [9, 8, 7, 6, 5]}
    d4 = pd.DataFrame(dl, index=['a', 'b', 'c', 'd', 'e'])  # 只给出关心的行列值，其他的会自动补齐
    print("d4: ", d4)

    data = {"城市": ['北京', '上海', '广州' ,'深圳', '沈阳'],
            "环比": [101.5, 101.2, 101.3, 102.0, 100.1],
            "同比": [120.7, 127.3, 119.4, 140.9, 130.4]}
    d5 = pd.DataFrame(data, index=['c1', 'c2', 'c3', 'c4', 'c5'])
    print("d5['环比']: ", d5['环比'])   # 获得列
    print("d5.ix['c1']: ", d5.ix['c1'])  # 获得行
    print("d5.ix[0]: ", d5.ix[0])  # 获得行
    print("d5['城市']['c1']: ", d5['城市']['c1'])  # 获得元素，先列后行
    print("d5.ix['c1']['城市']: ", d5.ix['c1']['城市'])  # 获得元素，先行后列

    # 增加或重排，重新索引
    d6 = d5.reindex(index=['c4', 'c1', 'c2', 'c3', 'c5'])
    d7 = d6.reindex(columns=['环比', '同比', '城市'])
    print("reindex: ", d6)
    print("reindex: ", d7)

    newrow = d7.index.insert(5, 'c6')
    newcol = d7.columns.insert(3, '新增')
    d8 = d7.reindex(columns=newcol, fill_value=0)
    d9 = d8.reindex(index=newrow, fill_value=0)
    print("d9.index: ", d9.index)  # index类型，不可修改的对象类型
    print("d9.columns: ", d9.columns)  # index类型，不可修改的对象类型

    newrow = d9.index.delete(5)
    newcol = d9.columns.insert(4, '测试')
    d10 = d9.reindex(index=newrow, fill_value=0)
    d11 = d10.reindex(columns=newcol, fill_value=0)
    print("d11: ", d11)

    # 删除，默认操作0轴
    d12 = d11.drop('c5')
    d13 = d12.drop('新增', axis=1)
    print("d13: ", d13)

    # 算术运算
    num = 5
    s = pd.Series([9, 8, 7, 6])
    s2 = pd.Series([9, 8, 7, 6, 5, 4])
    d = d5[['环比', '同比']]
    e = pd.Series([1, 2], index=['c1', 'c2'])
    a = pd.DataFrame({"环比": [3, 4], 'two': [1, 2]})
    a.index = ['c1', 'c2']

    print("s * num: ", s * num)  # 一维和零维间为广播运算，0维的数跟1维的每一个元素进行运算
    # 二维和一维、一维和零维间为广播运算
    print("s - s2: ", s - s2)  # 一维间按索引运算，补齐时缺项填充NaN
    print("a+d: ", a+d)  # 两维间按索引运算，补齐时缺项填充NaN
    print("a.add(d):　", a.add(d, fill_value=0))
    print("a.sub(d):　", a.sub(d, fill_value=0))
    print("a.mul(d):　", a.mul(d, fill_value=0))
    print("a.div(d+1):　", a.div(d+1, fill_value=0))

    # 比较运算
    # 同维度运算，要求尺寸一致，不存在填充，如果尺寸
    # 不同维度运算，广播运算，默认在1轴
    # a > b
    # a == b

    # 数据排序
    # 数据摘要，有损的提取数据特征的过程
    # 基本统计（含排序）、分布/累计统计、数据特征、相关性、周期性等、数据挖掘（形成知识）
    d.sort_index(axis=0, ascending=True)  # 对0轴的索引进行排序
    d.sort_index(axis=1, ascending=True)  # 对1轴的索引进行排序
    d.sort_values('环比', axis=0, ascending=False)  # 按某一列的值排序，NaN统一放到末尾
    d.sort_values('c1', axis=1, ascending=False)  # 按某一列的值排序，NaN统一放到末尾




if __name__ == '__main__':
    usePandasSeries()
    usePandasDataFrame()


# Pandas
# 提供方便的数据类型, 提供分析函数和分析工具
# 扩展数据类型，关注数据的应用表达，数据与索引间关系
# 基础数据类型，关注数据的结构表达，维度：数据间关系

# series类型
# 一维的带标签的数组
# 基本操作类似ndarray和字典，根据索引对齐

# dataframe类型
# 索引加多列数据组成，多列数据共用同样的索引，就是一个表格，每列值的类型可以不同
# 纵向索引叫index，0轴，axis=0，列索引
# 横向索引叫column，1轴，axis=1，行索引
# 常用二维数据，也可以表达多维数据

# 操作索引即操作数据，把变量当成单一的元素去考虑





