# -*- coding: UTF-8 -*-
import os
import time as t
import random as r
import os.path as op


# 打印开局信息
def printStart():
    start = t.perf_counter()
    print("Start ... at {}".format(t.strftime('%H:%M:%S %Y-%m-%d', t.gmtime())))
    # print("这个程序要对两个选手进行体育竞技模拟")
    return(start)


# 输入比赛的设定值
def getInputs():
    a = eval(input("请输入选手A的能力值(0-1): "))
    b = eval(input("请输入选手B的能力值(0-1): "))
    n = eval(input("请输入比赛的场次值: "))
    return(a, b, n)


# 逆推
def getInputs2():
    a = eval(input("请输入选手A的获胜次数(1-N): "))
    b = eval(input("请输入选手B的获胜次数(1-N): "))
    base = eval(input("请输入选手的基础能力值: "))
    return(a, b, base)


# 模拟n次比赛
def gameSim(n, probA, probB):
    winsA, winsB = 0, 0
    for i in range(n):
        scoreA, scoreB = gameSimOne(probA, probB)
        if scoreA > scoreB:
            winsA += 1
        else:
            winsB += 1
    return(winsA, winsB)


# 逆推
def gameSim2(winsA, winsB, base):
    probA, probB = base, base
    n = winsA + winsB

    while probA<=1 and probB<=1:
        winsAt, winsBt = gameSim(n, probA, probB)
        gap = (winsAt - winsA) / n
        if gap > 0.01:
            probB += 0.01
        elif gap < -0.01:
            probA += 0.01
        else:
            return(probA, probB)


# 模拟1次比赛
def gameSimOne(probA, probB):
    scoreA, scoreB = 0, 0
    serving = 'A'
    while not gameOver(scoreA, scoreB):
        if serving == 'A':
            if r.random() < probA:
                scoreA += 1
            else:
                serving = 'B'
        if serving  == 'B':
            if r.random() < probB:
                scoreB += 1
            else:
                serving = 'A'
    return(scoreA, scoreB)


# 判断游戏结束
def gameOver(a, b):
    return(a==15 or b == 15)


# 输出比赛结果
def output(winsA, winsB):
    n = winsA + winsB
    print("总比赛场次: {}".format(n))
    print("A获胜{}场，获胜比例为{:.2f}%".format(winsA, winsA/n*100))
    print("B获胜{}场，获胜比例为{:.2f}%".format(winsB, winsB/n*100))


# 逆推
def output2(probA, probB):
    print("选手A的能力值为: {:.2f}".format(probA))
    print("选手B的能力值为: {:.2f}".format(probB))


# 输入结尾信息
def printEnd(start):
    prog = t.perf_counter() - start
    print("End ... at {}".format(t.strftime('%H:%M:%S %Y-%m-%d', t.gmtime())))
    print("Total time: {:.2f}s".format(prog))


# 整数求和
def sum100():
    sum = 0
    for i in range(100):
        sum += i+1
    print(sum)


# 使用os.path库，路径操作，进程管理，环境参数
def useOS():
    # 路径操作
    print("abspath:", op.abspath('data/data.csv'))  # 绝对路径
    print("relpath:", op.relpath('/public/noncode/users/wuyang/myRepo/testPython/data/data.csv'))  # 相对路径
    print("normpath:", op.normpath('data//data.csv'))  # 归一化路径，统一格式，便于字符串处理
    print("join:", op.join("/public/noncode/users/wuyang/myRepo/testPython/data", "data.csv"))  # 组合路径
    print("dirname:", op.dirname('/public/noncode/users/wuyang/myRepo/testPython/data/data.csv'))  # 返回目录名称
    print("basename:", op.basename('/public/noncode/users/wuyang/myRepo/testPython/data/data.csv'))  # 返回文件名称
    print("exists:", op.exists("data/data.csv"))  # 判断文件或目录是否存在
    print("isfile:", op.isfile("data/data.csv"))  # 判断是否为文件，且存在
    print("isdir:", op.isdir("data/data.csv"))  # 判断是否为目录，且存在
    print("getsize:", op.isdir("data/data.csv"))  # 判断是否为目录，且存在
    print("getatime:", t.ctime(op.getatime("data/data.csv")))  # 返回文件或目录上一次的访问时间
    print("getmtime:", t.ctime(op.getmtime("data/data.csv"))) # 返回文件或目录上一次的修改时间
    print("getctime:", t.ctime(op.getctime("data/data.csv")))  # 返回文件或目录的创建时间

    # 进程管理
    print("pwd:", os.system("pwd >pwd.txt"))  # 执行程序或命令，只返回调用成功与否的返回值

    # 环境参数
    os.chdir("data")  # 修改当前程序操作的路径
    print("getcwd:", os.getcwd())  # 返回程序的当前路径
    print("getlogin:", os.getlogin())  # 获得当前系统的登录用户名称
    print("cpu_count:", os.cpu_count())  # 获得当前系统的CPU数量
    print("urandom:", os.urandom(5))  # 生成n个字节长度的随机字符串，通常用于加解密运算


# 自动安装脚本
def autoInstall():
    libs = {"numpy", "matplotlib"}
    try:
        for lib in libs:
            os.system("pip install " + lib)
        print("Successful")
    except:
        print("Failed somehow")



if __name__ == '__main__':
    # 体育竞技分析
    mode = eval(input("请输入游戏模式：求次数(1) or 求能力值(2): "))
    if mode == 1:  # 已知能力值，推获胜次数
        start = printStart()
        probA, probB, n = getInputs()
        winsA, winsB = gameSim(n, probA, probB)
        output(winsA, winsB)
        printEnd(start)
    elif mode == 2:  # 已知获胜次数，推能力值
        start = printStart()
        winsA, winsB, baseline = getInputs2()
        probA, probB = gameSim2(winsA, winsB, baseline)
        output2(probA, probB)
        printEnd(start)
    else:
        pass

    # 求和
    sum100()

    # 量化分析
    # 利用历史数据，机器学习，预测未来的涨跌，利用计算机完成自动交易，美国目前>60%的交易，计算机完成的

    # 操作系统交互
    useOS()

    # 批量安装第三方库
    autoInstall()




# 自顶向下，系统思维的简化版本，分而治之，设计
# 自底向上，模块化集成，执行

# 计算思维与程序设计
    # 逻辑思维，推理与演译为特征，数学为代表，形成公式，获得结果
    # 实证思维，实验和验证为特征，物理为代表，获得数据，获得结果
    # 计算思维，设计和构造为特点，计算机为代表（第三种人类思维特征），模拟运算的过程，利用计算机，获得结果
        # 基本特征是抽象和自动化，抽象是问题的计算过程，利用计算来自动化求解，是在出现比人具有更强大算力的设备之后出现的
        # 基于计算机强大的算力和海量的数据，抽象计算过程，不关注因果关系，而是关注事物的过程、设计和构造
        # 计算机程序设计为实现计算思维的主要手段，编程是将计算思维变成现实的手段
        # 既需要设计和构造为特征的抽象问题的能力，也需要以编程为手段的自动化执行的方法，才能将思维变成真正的结果

# 计算生态与Python语言
    # 1983年，MIT教授，GNU项目，想集中开发一个操作系统，大教堂模式
    # 1989年，GNU通用许可协议，自由软件时代到来
    # 1991年，Linux内核发布，操作系统的极小内核，放到网上，大批人polish，集市模式（真正推动信息技术发展的主要模式）
    # 1998年，Mozilla开源，产生Firefox等开源浏览器，标志着开源生态的建立
    # 计算生态，以开源项目为组织形式，竞争发展、相互依存、迅速迭代，信息技术的更新换代，技术的演化路径
    # Python，13万第三方库，野蛮生长、自然选择，学生、工程师、科学家，只要愿意贡献就可以，同一个功能，二个以上库相互竞争
    # 一个库建立在其他库的基础上，逐级封装
    # request库，爬虫界
    # numpy库搭好了高速的，pandas库，matplotlib库在此基础上做
    # AlphaGo深度学习算法采用python语言开源
    # API，应用程序编写接口，跟生态是不同的，顶层设计形成
    # 创新：跟随创新、集成创新、原始创新，创新驱动发展，加速科技类应用创新的重要支撑，发展科技产品商业价值的重要模式，国家科技体系安全和稳固的基础
    # 计算生态的应用，编程的起点，不是刀耕火种，而是站在巨人的肩膀上，编程的起点不是算法，而是系统，编程如同搭积木，利用计算生态为主要模式
    # 编程的目的是为了解决问题，编程的目标是快速解决问题
    # http://python123.io，优质第三方库，看见更大的世界，遇见更好的自己

# 用户体检与软件产品
    # 用户体检是用户对产品的主观感受和认识
    # 方法一，进度展示
    # 方法二，异常处理，合规性检查
    # 方法三，打印输出，过程信息，了解运行状态
    # 方法四，log文件
    # 方法五，打印帮助信息

# 基本的程序设计模式
    # IPO，明确计算部分及功能边界，编写程序，调试程序
    # 自顶向上
    # 模块化设计，封装，分而治之
    # 松耦合，紧耦合，看数据交互
    # 配置化设计，数据和程序分离，程序引擎+配置文件，程序执行和配置分离，关键在于接口设计，清晰明了，灵活可扩展

# 应用开发
    # 产品定义，对应用需求充分理解和明确定义，考虑商业模式
    # 系统架构设计，以系统方式思考产品的技术实现，包括系统架构、关注数据流、模块化、体系架构
    # 设计与实现，结合架构完成关键设计及系统实现，结合可扩展性、灵活性等进行设计优化
    # 用户体验，从用户角度思考应用效果，用户至上，体检优先，以用户为中心

# 看见更大的python世界
    # https://pypi.org, python package index
    # PSF (python software foundation)维护的展示全球python计算生态的主站
    # 安装python第三方库，pip安装（主要方式，需要联网）
        # pip install <第三方库>       # 安装
        # pip install -U <第三方库名>  # 更新
        # pip uninstall <第三方库名>   # 卸载
        # pip download <第三方库名>    # 下载
        # pip show <第三方库名>        # 列信息
        # pip search <第三方库名>      # 查找
        # pip list                     # 列出已安装的库
    # 安装python第三方库，集成安装
        # 结合特定的python开发工具，批量的安装一批库
        # anaconda,https://www.continuum.io
        # 支持近800个第三方库，比较主流，适合数据领域
    # 安装python第三方库，文件安装
        # 某些第三方库，pip下载后，需要编译再安装
        # 如果只提供源代码，没提供编译后的文章，需要依赖系统的编译环境
        # UCI页面，https://www.lfd.uci.edu/~gohlke/pythonlibs/
        # 先下载文件，再用pip install <文件名>进行安装