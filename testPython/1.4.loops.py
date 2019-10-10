# -*- coding: UTF-8 -*-
import time as t
import turtle as tt
import random as r



# 程序高级异常处理
def input_num(message):
    try:
        num = eval(input(message))  # 首先，试运行try对应的语句块1
    except NameError:
        print("Please input a int number")  # 如果，语句块1出现异常，则执行except所对应的语句块2
    else:
        return(num)  # 如果，语句块1未出现异常，则执行else所对应的语句块3
    finally:
        pass  # 最后，不管语句块1是否出现异常，均执行finally所对应的语句块4


# input两个变量，逗号分隔，一次接受两个或多个输入变量
def input_nums(message):
    try:
        height, weight = eval(input(message))
    except NameError:
        print("Please input two numbers")
    else:
        return(height, weight)
    finally:
        pass


# 二分叉结构的紧凑形式，是表达式，而不是语句，不能赋值
def compact_branch():
    num = input_num("输入一个整数, >5，对，反之，错： ")
    print("{}".format("对" if (num>5) else "错"))


# 多分叉结构，判断成绩，一定要注意多条件之间有包含关系，注意变量取值范围的覆盖关系
def multiple_branch():
    score = input_num("输入一个学生成绩，返回等级：")
    grade = ''
    if score>=90:
        grade = 'A'
    elif score>=80:
        grade = 'B'
    elif score>=60:
        grade = 'C'
    else:
        grade = 'D'
    print("Grade is {}".format(grade))


# 组合条件, and, or, not, 例如not True
def combine_conditions():
    num = input_num("多重条件判断，请输入一个整数：")
    if num>99 or num<1:
        print("Wrong")
    else:
        print("Right")


# 计算身体的BMI
def calculate_BMI():
    height, weight = input_nums("输入两个数值（身高/cm，体重/kg），逗号隔开：")
    bmi = weight/pow(height/100, 2)  #  china_cutoff = [18.5, 24, 28], international_cutoff = [18.5, 25, 30]

    if bmi < 18.5:
        type = "{}+{}".format("thin", "thin")
    elif bmi < 24:
        type = "{}+{}".format("proper", "proper")
    elif bmi < 25:
        type = "{}+{}".format("overweight", "proper")
    elif bmi < 28:
        type = "{}+{}".format("overweight", "overweight")
    elif bmi < 30:
        type = "{}+{}".format("fat", "overweight")
    else:
        type = "{}+{}".format("fat", "fat")

    print("BMI={:.1f}\nChina type={}\nInternational type={}".format(bmi, type.split('+')[0], type.split('+')[-1]))


# 遍历循环
def for_loop():
    # 数字遍历循环，计数循环
    print("\n>>>>读数循环")
    M, N, K = eval(input("3 numbers seperated by comma: "))
    for i in range(M, N, K):
        print(i)

    # 字符串遍历循环
    print("\n>>>>字符串遍历循环")
    s = input("Input a string:")
    for c in s:
        print(c, end=",")  # 不换行，每个字符后面加逗号

    # 列表遍历循环
    print("\n\n>>>>列表遍历循环")
    for i in [123, 'PY', 456]:
        print(i, type(i), end=" ~ ")

    # 文件遍历循环
    print("\n\n>>>>文件遍历循环")
    fh = open('data/input.txt')
    for line in fh:
        print(line, end="")


# 无限循环
def while_loop():
    a = input_num("Please input a number: ")
    while a > 3:
        a = a-1
        print('Output: ', a)


# 中断循环
def stop_loop():
    print("\ncontinue:")
    for i in 'python':
        if i == 't':
            continue  # 本次循环不执行
        print(i, end="")

    print("\n\nbreak:")
    for i in 'python':
        if i == 't':
            break  # 循环跳出，一个break只能跳出一层循环
        print(i, end="")


# 高级循环，else，如果没有被break退出，正常完成循环的奖励，类似异常处理的else
def loop_else():
    c = input("Input a char: ")

    # 无论python字符串中是否含有c字条，因为没有break，循环执行结束之后，都会打印finished
    print("\nfor_else continue")
    for i in 'python':
        if i == c:
            continue
        print(i, end="")
    else:
        print("\nFinished!")

    # 如果python字条串中含有c字符，则break，无法打印finished
    print("\nfor_else break")
    for i in 'python':
        if i == c:
            break
        print(i, end="")
    else:
        print("\nFinished!")


# 梅森旋转算法，根据种子产生一系列随机序列，每一个数就是一个随机数，产生确定的伪随机数
def random_num():
    r.seed(1)  # 参数默认为当前系统时间，设置seed，可以复现程序运行过程
    len = 5
    print("\n0-1之间的随机小数")  # 精度为小数点后16位
    for i in range(len):
        print(r.random())

    print("\n[a, b]之间的随机小数")
    for i in range(len):
        print(r.uniform(15, 16))

    print("\n[a, b]之间的随机整数")
    for i in range(len):
        print(r.randint(1, 100))

    print("\n[m, n]之间的随机整数，步长为k")
    for i in range(len):
        print(r.randrange(0, 100, 10))

    s = input("\n从字符串中随机选择字符 ")
    for i in range(len):
        print(r.choice(s))

    print("\n对列表进行打乱重排")  # input是一个list
    for i in range(len):
        s = list(s)
        r.shuffle(s); print(s)  # 多行放一起，分号分隔，但不建议

    print("\nk比特长的随机整数")
    for i in range(len):
        print(r.getrandbits(16))


# 圆周率的近似计算公式
def calc_pi():
    pi = 0
    N = 100
    for k in range(N):
        pi += (4/(8*k+1) - 2/(8*k+4) - 1/(8*k+5) - 1/(8*k+6)) / pow(16, k)
    print("pi={}".format(pi))


# 圆周率的数值模拟计算，蒙特卡罗方法，计算撒点数量的比值，数学思维，公式求解，前提是有数学规则
def calc_pi_MTCL():
    # point_num = 1000 * 1000  # 总点数
    point_num = input_num("请选择要撒点的总数：")
    hits = 0  # 圆内的点数
    start = t.perf_counter()
    r.seed(10)

    for i in range(point_num):
        x, y = r.random(), r.random()
        dist = pow(x**2 + y**2, 0.5)
        if dist<=1.0:
            hits += 1
        pi = hits / point_num * 4  # 撒点是在右上象限，是圆的1/4
    print("pi={}, time={:.5f} s".format(pi, t.perf_counter()-start))


# 把蒙特卡罗方法计算圆周率进行可视化，计算思维，用计算机自动化求解，抽象了一种过程
def plot_pi_MTCL():
    point_num = input_num("请选择要撒的总点数：")
    start = t.perf_counter()
    size = 2
    r.seed(10)
    tt.setup()

    for i in range(point_num):
        x, y = r.random(), r.random()
        dist = pow(x**2 + y**2, 0.5)
        tt.penup()
        tt.goto(x*100, y*100)
        tt.pendown()
        if dist<=1.0:  # 在圆内
            tt.dot(size, 'blue')
        else:  # 在圆外，正方形里
            tt.dot(size, 'black')


if __name__ == '__main__':
    # compact_branch()
    # multiple_branch()
    # combine_conditions()

    # while True:
    #     calculate_BMI()

    # for_loop()
    # while_loop()
    # stop_loop()
    # loop_else()

    # random_num()
    while True:
        # calc_pi()
        calc_pi_MTCL()
        # plot_pi_MTCL()

