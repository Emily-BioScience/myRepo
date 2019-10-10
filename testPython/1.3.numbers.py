# -*- coding: UTF-8 -*-
# UTF-8的注释，是为了兼容中文

# 进制转换
def num_convert(num):
    i2 = bin(num)
    i8 = oct(num)
    i10 = int(num)
    i16 = hex(num)

    print("\n进制转换")
    print(num, "=>", i2)  # 0b1111101000
    print(num, "=>", i8)  # 0o1750
    print(num, "=>", i10)  # 1000
    print(num, "=>", i16)  # 0x3e8


# 数字和字符串转换
def num_string_convet(num):
    string =str(num)
    num_real = float(num)
    num_complex = complex(num)
    num_convert = int(string)  # 变成整数，并舍弃小数部分

    print("\n数字和字符串转换")
    print(string, ':', type(string))
    print(num_real, ":", type(num_real))
    print(num_complex, ":", type(num_complex))
    print(num_convert, ':', type(num_convert))


#  数学运算
def math_convert(a, b):
    print("\n数学运算")
    print('a+b:', a+b)
    print('a-b:', a-b)
    print('a*b:', a*b)
    print('a/b:', a/b)
    print('a//b:', a//b)  # 整除
    print('a%b:', a%b)  # 取余，模运算
    print('a**b:', a**b)  # 幂操作
    print('+a:', +a)  # x本身
    print('-b:', -b)  # 负数
    # # 赋值运算符
    # a *= b
    # print(a)
    # print(b)


# 数值运算函数
def math_function():
    print("\n数值运算函数")
    print("abs(-5) =>", abs(-5))  #
    print("divmod(10, 3) =>", divmod(10, 3))
    print("pow(10, 3) =>", pow(10, 3))
    print("round(10.1113, 2) =>", round(10.1113, 2))
    print("max(1,9, 3, 4, 5) =>", max(1, 9, 3, 4))


# 天天向上
def daychange(rate):
    dayup = pow(1+rate, 365)
    daydown = pow(1-rate, 365)
    return([dayup, daydown])


# 天天向上，工作日模式
def daychange_haveWeekend(rate):
    daychange = 1
    for i in range(365):
        if i%7 in [0, 6]: # weekend
            daychange *= (1)
        else:  # working
            daychange *= (1+rate)
    return(daychange)



if __name__ == "__main__":
    num_convert(1000)
    # num_string_convet(1000)
    # math_convert(10, 3)
    # math_function()

    rates = [0.001, 0.005, 0.01, 0.019]
    for rate in rates:
        change = daychange(rate)
        change_loop = daychange_haveWeekend(rate)
        print("rate: {:.5f} => dayup: {:.3f}".format(rate, change[0]))
        print("              => daydown: {:.3f}".format(change[1]))
        print("              => daychange: {:.3f}".format(change_loop))


    rate=0.0100
    change=37.78
    while rate:
        change_haveweekend = daychange_haveWeekend(rate)
        if change_haveweekend >= change:
            print("rate={:.3f}, change={:.3f}".format(rate, change_haveweekend))
            break
        else:
            rate = rate + 0.001