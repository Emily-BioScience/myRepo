# -*- coding UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


def usePyplot(outfile):
    # matplotlib.rcParams['font.family'] = 'wqy-zenhei'  # 中文支持，wqy-zenhei
    # matplotlib.rcParams['font.style'] = 'italic'  # normal，控制字体网格
    # matplotlib.rcParams['font.family'] = 'large'  # 控制字体大小，整数字号，或者large, x-small

    plt.plot([1, 1, 4, 5, 2], [3, 5, 6, 7, 8])  # 先x轴，后y轴，按照列表顺序绘制
    plt.ylabel('grade')
    plt.axis([-1, 10, 0, 8])  # x轴起始于-1， 终止于10， y轴起始于0， 终止于6
    plt.savefig(outfile, dpi=600)  # PNG文件

    a = np.arange(0, 5, 0.02)
    plt.subplot(3, 2, 1)  # 绘图区域分割成3行，2列（1-6，左上，右上，左中，右中，左下，右下），创建一个分区体系，定位到一个子区域
    plt.ylabel('test')
    plt.grid(True)
    plt.plot(a, f(a))

    plt.subplot(322)  # 绘图区域分割和定位，可省略逗号
    plt.grid(True)
    plt.plot(a, np.cos(2*np.pi*a), 'r--')  # 虚线
    plt.title('haha')
    plt.xlabel('test2')

    plt.subplot(323)
    a = np.arange(10)
    plt.grid(True)
    # plt.axis(-1, 6, 0, 20)  # 明确x, y轴的坐标范围
    plt.plot(a, a*1.5, 'go-', a, a*2.5, 'rx', a, a*3.5, '*', a, a*4.5, 'b-.')  # format_string，颜色字符、风格字符和标记字符组合使用
    plt.ylabel('test3')
    plt.text(2, 30, r'$\mu=100$', fontsize=15)  # 引入latex格式文本
    plt.annotate(r'$\mu=100$', xy=(3, 10), xytext=(2, 25), arrowprops = dict(facecolor='black', shrink=0.1, width=1))  # 增加带箭头的注释文本
    outfile2 = outfile.replace('.out', '.out2')
    plt.savefig(outfile2)

    plt.subplot2grid((3, 3),(2, 0), colspan=2)
    plt.plot(a, a*2)
    outfile3 = outfile.replace('.out', '.out3')
    plt.savefig(outfile3)

    # 自定义绘图区域
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :-1])
    ax3 = plt.subplot(gs[1:, -1])
    ax4 = plt.subplot(gs[2, 0])
    ax5 = plt.subplot(gs[2, 1])
    plt.show()
    outfile3 = outfile.replace('.out', '.out3')
    plt.savefig(outfile3)

    # 饼图
    labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
    sizes = [15, 30, 45, 10]
    explode = [0, 0.1, 0, 0]
    plt.pie(sizes, explode=explode, labels=labels, shadow=False, startangle=0)
    plt.axis('equal')
    outfile4 = outfile.replace('.out', '.out4')
    plt.savefig(outfile4)

    # 直方图
    plt.subplot(2, 1, 1)
    np.random.seed(0)
    mu, sigma = 200, 20
    a = np.random.normal(mu, sigma, size=100)
    plt.hist(a, 20, normed=1, histtype='stepfilled', facecolor='b', alpha=0.75)  # bin为20，直方图的个数
    plt.title('Normal distribution')
    outfile5 = outfile.replace('.out', '.out5')
    plt.savefig(outfile5)
    plt.show()

    # 极坐标图
    N = 20
    theta = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    radii = 10 * np.random.rand(N)
    width = np.pi / 4 * np.random.rand(N)

    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta, radii, width=width, bottom = 0.0)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.viridis(r/10.0))
        bar.set_alpha(0.5)
    plt.show()

    # 散点图
    fig, ax = plt.subplots()
    ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o')
    ax.set_title('Scatter plot')
    plt.show()



if __name__ == '__main__':
    usePyplot('output/2.2.matplotlib.out.jpg')


# plt.plot(x, y, format_string, args)
# format_string: 颜色
#   'b': 蓝色
#   'g': 绿色
#   'r': 红色
#   'c': 青绿色，cyan
#   'm': 洋红色，magenta
#   'y': 黄色
#   'k': 黑色
#   'w': 白色
#   '0.8':  灰度值字符串
#   '#008000':  RGB某颜色
# format_string: 标记字符
#   '.': 点标记
#   ',': 像素标记（极小点）
#   'o': 实心圈标记
#   'v': 倒三角标记
#   '^': 上三角标记
#   '>': 右三角标记
#   '<': 左三角标记
#   '1': 下花三角标记
#   '2': 上花三角标记
#   '3': 左花三角标记
#   '4': 右花三角标记
#   's': 实心方形标记
#   'p': 实心五角标记
#   '*': 星形标记
#   'h': 竖门边形标记
#   'H': 横六边形标记
#   '+': 十字标记
#   'x': x标记
#   'D': 菱形标记
#   'd': 瘦菱形标记
#   '|': 垂直线标记
# format_string: 风格字符
#   '-': 实线
#   '--': 破折线
#   '-.': 点划线
#   ':': 虚线
#   '' / ' ': 无线条（空或空格）
# **kwargs:
# color='green'
# linestyle='dashed'
# marker='o'
# markerfacecolor = 'blue'
# markersize = 20
