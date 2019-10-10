# -*- coding: UTF-8 -*-
import jieba
import turtle as t
import wordcloud as w
from scipy.misc import imread


# 读、写文件，open(), close()，测试
def read_and_write(infile,outfile):
    # 写文件
    try:
        fo = open(outfile, 'x')  # 创建写模式，如果文件不存在，则创建，如果文件存在，汇报FileExistsError
    except FileExistsError:
        print("write to {}:  failed".format(outfile))

    fo = open(outfile, 'w')  # 覆盖写模式，如果文件不存在，则创建，如果文件存在，则覆盖原文件，不会汇报异常，常用！
    # fo = open(outfile, 'a')  # 追加写模式，如果文件不存在，则创建，如果文件存在，则在文件末尾追加内容，不会汇报异常
    # fo = open(outfile, 'a+')  # 追加写模式+读文件，即能读，又能追加写

    # 读文件
    try:
        fh = open(infile)  # 默认值是rt，即只读模式，文本形式打开，如果文件不存在，则返回FileNotFoundError
        # fh = open(infile, 'rt')  # 文本形式读入，只读模式
        # fh = open(infile, 'rb')  # 二进制形式读入，只读模式
        # fh = open(infile, 'b')   # 二进制形式读入，只读模式，r是默认
    except FileNotFoundError:
        print("{} not exists\n".format(infile))
    else:
        # print(type(fh.read()))  # 整个文件读入，存入字符串
        fo.write("1. read size=8： {}===".format(fh.read(8)))  # 参数是size，读入长度为10的前面的字符串，换行符算进去
        fo.write("2. read line=3: {}===".format(fh.readline(3)))  # 读入一行,参数是size，读入第一行的前3个字符
        fo.write("2. read line: {}===".format(fh.readline()))  # 读入一行,带行末换行符
        for line in fh.readlines(1):  # 按行读取，读取多行，带行末换行符,每行是一个字符串，弄成一个列表，参数是行数，读入第一行
            fo.write("3. read lines: {}===".format(line))

        # fo.write("test\n")  # 与read()相对应，写入字符串
        fo.writelines(["lili\n", "hahatest\n"])  # 与writelines相对应，写入多行

        fh.close()  # 使用打开时赋予的文件句柄，进行关闭，如果忘写了，python解释器退出时，会自动关闭
        fo.close()


# 逐行遍历，读写的推荐方式
def read_and_write_recommended(infile, outfile):
    try:
        fh = open(infile)
    except:
        print("Open file failed:{}".format(infile))
    else:
        fo = open(outfile, 'w')
        for line in fh:  # 分行读入，逐行处理  #### 最推荐的读文件方式 ！！！！！！
            fo.write(line)  # 写入字符串，或字节流
            fo.writelines([line, line])   # 并不分行，直接拼接，只是把元素全为字符串的列表写进去
        fo.seek(0)  # 改变文件操作指针的位置，0表示，回到文件开头
        fo.write("start")
        fh.close()
        # for line in fh.readlines():  # 一次读入，分行处理
        #     print(line, end="")


# 自动轨迹绘制，数据和功能进行分离，进行数据驱动的自动运行
def auto_trace(infile):
    try:
        fh = open(infile)
    except:
        print("Open file failed: {}".format(infile))
    else:
        start_plot()
        for line in fh:
            auto_plot(line.strip())
        fh.close()
        close_plot()


# 根据数据接口，绘制每一条线，接口化设计，格式化设计接口，清晰明了
def auto_plot(control):
    try:
        fd, turn, angle, r, g, b = list(map(eval, control.split(","))) # map内嵌函数，第一个参数的功能作用于第二个参数的每一个元素，第一个参数是函数的名字，第二个参数是一个迭代类型
    except:
        print("input 6 numbers, separated by commas")
    else:
        # print(fd, turn, angle, r, g, b)
        t.pencolor(r, g, b)
        t.fd(fd)
        if turn == 0:
            t.left(angle)
        elif turn == 1:
            t.right(angle)


# 开启turtle绘图，
def start_plot():
    t.title("自动轨迹绘制"); t.setup(800, 600)  # 设置窗体大小
    t.pencolor('red'); t.pensize(5)
    t.penup(); t.goto(0, 0); t.pendown()
    t.colormode(1.0)  # 1.0, 255


# turtle绘图框关闭
def close_plot():
    t.hideturtle()
    t.exitonclick()


# 一维数据处理，列表，数组，集合，存储、表示、操作
def one_dim_data(outfile):
    ls = ['一', '维', '有', '序', '数', '据']
    for item in ls:
        print("list item", item)

    st = {'一', '维', '无', '序', '数', '据'}
    for item in st:
        print("set item:", item)

    fo = open(outfile, 'w+')
    fo.writelines(["$".join(ls), "\n"])  # 缺点是，数据中无法存在分隔符
    fo.writelines(["$".join(st), "\n"])  # 数据中无法存在分隔符，规避方法：用特殊字符

    fo.seek(0)
    txt = fo.read()
    ls = txt.split("$")
    for item in ls:
        print("output:", item)
    fo.close()


# 二维数据处理，多个一维数据构成，是一维数据的组合形式，表格是典型的二维数据，表头可作为数据的一部分，或者单拎出去
def two_dim_data(infile):
    try:
        fh = open(infile)
    except:
        print("File not found: {}".format(infile))
    else:
        data = []
        for line in fh:
            items = line.strip().split("$")
            data.append(items)

        count = 0
        for ref in data:
            count += 1
            for c in ref:
                print(count, ":", c)

    # 二维列表表达二维数据，是最基本的数据方式
    # 处理大数据，可以用numpy, pandas等框架来进行


# 多维数据处理，由一维或二维数据，在新的数据维度上进行的扩展
def multi_dim_data():
    identity = {
        'first name':'Yang',
        'last name' : 'Wu',
        'address':{
            'city':'Beijing',
            'zipcode':'100190'
            },
        'major':['bioinformatics', 'biotechnology']
        }
    print("My name is {} {}.".format(identity['first name'], identity['last name']))
    print("I live in {}, with a zipcode of {}.".format(identity['address']['city'], identity['address']['zipcode']))
    print("My major is {}.".format(" and ".join(identity['major'])))


# 使用csv格式文件，如果值缺失，逗号不能少，如果数据里包含逗号，则加转义符，或者双逗号，看不同的软件怎么处理
# 一般索引习惯，ls[row][column]，先行后列，找交叉点，外层列表每个元素是一行
def use_csv(infile, outfile):
    try:
        fh = open(infile)
    except:
        print("File not found: {}".format(infile))
    else:
        ls = []
        for line in fh:
            ls.append(line.strip().split(','))
        fh.close()

    fo = open(outfile, 'w')
    for item in ls:
        fo.write(",".join(item) + "\n")
    fo.close()


# 绘制词云
def use_word_cloud(infile, outfile):
    try:
        fh = open(infile)
    except:
        print("File not found: {}".format(infile))
    else:
        for line in fh:
            if '#' in line:
                pass
            else:
                data = " ".join(line.strip().split(','))
                print(data)
        fh.close()

        mk = imread('data/5.time.ico')
        wc = w.WordCloud(width=600, height=400, \
                         min_font_size=10, max_font_size=20, font_step=1, \
                         font_path="/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc", \
                         max_words = 20, \
                         background_color = 'white', \
                         mask = mk
                         )  # 窗口长宽，最大最小字号，字号步进间隔，字体用微软雅黑，最多显示20上词，背景颜色设为白色，指定词云形状为图形
        wc.generate(data)  # 用空格分隔单词，统计单词出现次数并过滤，根据统计配置字号，布局颜色环境尺寸
        wc.to_file(outfile)


# 绘制政府工作报告词云
def test_word_cloud(infile, outfile):
    try:
        fh = open(infile)
    except:
        print("File not found: {}".format(infile))
    else:
        txt = fh.read()
        fh.close()
        words = jieba.lcut(txt)
        data = " ".join(words)
        mask = imread('data/1.7.chinamap.png')

        wc = w.WordCloud(width=1000, height=700, \
                         font_path="/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc", \
                         background_color = 'white',mask = mask
                         )  # 窗口长宽，最大最小字号，字号步进间隔，字体用微软雅黑，最多显示20上词，背景颜色设为白色，指定词云形状为图形
        wc.generate(data)  # 用空格分隔单词，统计单词出现次数并过滤，根据统计配置字号，布局颜色环境尺寸
        wc.to_file(outfile)  # 进一步优化迭代



if __name__ == '__main__':
    # read_and_write('data/1.7.input.txt', 'output/1.7.output.txt')  # 绝对路径和相对路径都可以
    # read_and_write_recommended('data/1.7.input.txt', 'output/1.7.output.txt')  # 逐行遍历
    # auto_trace('data/1.7.trace.in')  #按照文件中的数据接口，进行自动图形绘
    # one_dim_data('output/1.7.output.txt')
    # two_dim_data('output/1.7.output.txt')
    # multi_dim_data()
    # use_csv('data/1.7.input.txt', 'output/1.7.output.txt')
    # use_word_cloud('data/1.7.input.txt', 'output/1.7.wordcloud.png')
    test_word_cloud('data/1.7.govWorkReport.txt', 'output/1.7.wordcloud.png')

