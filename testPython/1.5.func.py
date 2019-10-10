# -*- coding: UTF-8 -*-

import time as t
import turtle as tt
# import sys
# sys.setrecursionlimit(100000)  # 解决递归深度的问题


# 可选参数，定义时需要指定初值，按位置或名称传递
def func_optional(a, b=1):
    return(a*b)


# 可变参数，c可以是1个或者多个值，接收用户输入的多个输入
def func_multiple(a, b=1, *c):
    sum = a*b
    for i in c:
        sum *= i
    return(sum, a, b, c)


# 变量的作用域：
# 基本数据类型，无论是否重名，函数内为局部变量，函数外为全局变量，不是一个变量
#               可以通过global保留字在函数内部，声明全局变量，则可以在函数内部，使用外部的基本数据类型的全局变量
# 组合数据类型，如果局部变量未真实创建，直接使用的话，是在使用外部的全局变量，会改变其取值
#               如果局部变量重新创建了，则变成函数内部的局部变量，跟外部变量没有关系了
def variable_scope(x, y):
    global a   # 加入global保留字后，此变量a已变为全局全量，等同于main函数中的变量a
    a = x
    ls.append(y)  # 组合数据类型，未在函数内部定义，视为全局变量

def variable_scope2(x, y):
    global a   # 加入global保留字后，此变量a已变为全局全量，等同于main函数中的变量a
    a = x
    ls = []; ls.append(y)  # 组合数据类型，在函数内部真实的创建，视为局部变量


# 函数的参数，可选参数，可变参数，试验
def use_func_parameters():
    try:
        a, b = eval(input("输入多个数字："))
    except:
        print("error")
    else:
        print("a, b: ", func_optional(a, b))  # 函数返回一个值
        print("a: ", func_optional(a))
        all = func_multiple(a, b, 5, 6)  # 多个参数，通过可变参数c传入，存储为无组类型
        print("all: ", all)  # 函数返回多个值，存储在一个元组中
        print("sum: ", all[0])
        print("c: ", all[3])
    finally:
        pass


# 函数的作用域，全局变量和局部变量，基本数据类型和组合数据类型，试验
def use_func_scope():
    global ls
    ls = ['F', 'f']
    variable_scope('global', 'global')  # 组合数据类型是由指针来体现的，所以函数中未真实创建，使用的是外部全局变量的指针
    print("\na: ", a)
    print("list: ", ls)

    ls = ['F', 'f']
    variable_scope2('local', 'local')  # 函数中真实创建同名的组合数据类型，则指针改变，成为局部变量
    print("\na: ", a)
    print("list: ", ls)


# 使用lambda函数定义匿名函数
def use_lambda():
    f = lambda x: pow(x, 2)+ 2*x +1  # 运用lambda函数，来通过一行来定义函数
    for i in range(5):
        print(i, "=>", f(i))   # lambda函数不是函数定义的常用形式，主要作用是作为一些特殊函数或方法的参数


# 绘制每一个数字的起始
def plot_start(width=5, color='purple'):
    tt.setup(); tt.pencolor(color)  # 绘制开始
    tt.penup(); tt.seth(0); tt.pensize(3); tt.fd(width)  # 绘制间隔

# 绘制每一个字符
def plot_text(c, width=5, color='purple'):
    plot_start(width, color)
    tt.write(c, font=('Arial', width, "normal"))
    tt.penup()
    tt.fd(width*2)

# 获得每一个数字的指令
def control_num(num):
    controls = [[0, 1, 1, 1, 1, 1, 1], \
             [0, 1, 0, 0, 0, 0, 1], \
             [1, 0, 1, 1, 0, 1, 1], \
             [1, 1, 1, 0, 0, 1, 1], \
             [1, 1, 0, 0, 1, 0, 1], \
             [1, 1, 1, 0, 1, 1, 0], \
             [1, 1, 1, 1, 1, 1, 0], \
             [0, 1, 0, 0, 0, 1, 1], \
             [1, 1, 1, 1, 1, 1, 1], \
             [1, 1, 0, 0, 1, 1, 1]]
    return(controls[num])

# 绘制7段数码管的7段， turn right, turn down, turn left, turn up, turn up, turn right, turn down
def plot_seven_bubes(num, width=10, color='purple'):
    plot_start(width/2, color)  # 间隔：半个数字宽度
    control = control_num(num)   # 将数字转换为画图的指令

    for i in range(len(control)):  # 7部绘制，根据数学对应的绘制指令进行画图
        # set angle
        if i in [0, 5]: # go right
            tt.seth(0)
        elif i in [1, 2, 3, 6]:  # turn right
            tt.right(90)
        elif i == 4:  # go up
            tt.seth(90)

        # draw a line
        if control[i]:
            tt.pendown()
        else:
            tt.penup()
        tt.fd(width)  # 绘制：1个数学宽度

# 绘制时间的七段数码管
def test_seven_tubes():
    current = t.strftime('%Y-%m=%d+', t.gmtime())
    width = 20
    color = 'red'

    for c in current:
        if c == '-':
            plot_text('年', width, color)
            color = 'green'
        elif c == '=':
            plot_text('月', width, color)
            color = 'blue'
        elif c == '+':
            plot_text('日', width, color)
        else:
            plot_seven_bubes(int(c), width, color)  # 画每个数字

    tt.hideturtle()
    # tt.exitonclick()


# 字符串反转
def string_reverse(mystr):
    if len(mystr):
        revstr.append(mystr[-1])
        string_reverse(mystr[0:len(mystr)-1])
    else:
        return mystr


# 斐波那契数列
def feb_series(n):
    if n in [1, 2]:
        return(1)
    else:
        return(feb_series(n-1)+feb_series(n-2))


# 汉诺塔
def hano_tower(n, start, end, middle):
    global count
    if n == 1:
        print("{}: {} => {}".format(n, start, end))
        count += 1
    else:
        hano_tower(n-1, start, middle, end)
        print("{}: {} => {}".format(n, start, end))
        count += 1
        hano_tower(n-1, middle, end, start)


# 科赫雪花
def kehe_curve(size, n):
    if n==1:
        tt.pendown()
        tt.fd(size)
    else:
        for angle in [0, 60, -120, 60]:
            tt.left(angle)
            kehe_curve(size/3, n-1)


if __name__ == '__main__':
    # use_func_parameters()
    # use_func_scope()
    # use_lambda()
    print('test')
    print("测试")
    # test_seven_tubes()

    # mystr = input("")
    # print(mystr[::-1])  # 字符串反转

    # revstr = []
    # string_reverse(mystr)
    # print("".join(revstr))

    # for n in range(1, 10):
    #     print(n, feb_series(n))

    count = 0
    hano_tower(8, 'A', 'C', 'B')
    print("count={}".format(count))

    if 1:
        width = 100
        color = 'yellow'
        n = 3
        angle = 120
        plot_start(width, color)
        kehe_curve(width, n)
        tt.right(angle)
        kehe_curve(width, n)
        tt.right(angle)
        kehe_curve(width, n)
        tt.hideturtle()
        tt.exitonclick()








