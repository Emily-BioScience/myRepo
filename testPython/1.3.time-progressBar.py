# -*- coding: UTF-8 -*-

import time as t

# 获取时间
def gettime():
    # print(t.time())  # 获取系统的时间戳，系统内部的时间值，是浮点数，记录的是1970-1-1 00：00：00为起始的时间（秒）
    # print(t.ctime())   # 返回字符串，可读
    print(t.gmtime())  # 计算机可以处理的时间格式


# 时间格式化输出
def formattime():
    current = t.gmtime()   # 获得时间
    t_format = t.strftime('%Y-%m-%d %H:%M:%S', current)   # 格式化输出，控制符以%和字符表示
    t_time = t.strptime(t_format, '%Y-%m-%d %H:%M:%S')  # 将字符拆解
    print("Time is", t_format)
    print(t_time)


# 程序计时
def counttime():
    start = t.perf_counter()
    t.sleep(1) # 休眠60秒，可以是浮点数
    end = t.perf_counter()
    print('distance =', end-start)


# 文本进度条: 人机交互的纽带
def text_progressBar():
    scale = 20
    print("# {:=^30} #".format('执行开始'))
    start = t.perf_counter()
    for i in range(scale+1):
        ratio = (i/scale)*100
        finished = int(round(ratio)/10)
        left = 10 - finished
        prog = t.perf_counter() - start
        print("\r{:>3.0f} %  [{}=>{}] {:.2f}s".format(ratio, finished * "=", left * ".", prog), end="")   # end=""，不换行，“\r”，光标移动本行首
        t.sleep(0.5)  # 持续进度条
    print("\n# {:=^30} #".format('执行结束'))


# 设计不同的文本进度条
def text_progressBar_design():
    print(1)



if __name__ == "__main__":
    gettime()
    formattime()
    counttime()
    text_progressBar()
    text_progressBar_design()
